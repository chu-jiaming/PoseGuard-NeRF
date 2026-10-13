#!/usr/bin/env python
"""Paired raw/Sim(3) diagnostic on existing checkpoints. Never trains a model."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poseguard.alignment import fit_sim3, transform_poses
from scripts.sv100_pipeline import GT


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def jobs():
    with (ROOT/'results/SV100_full_summary.csv').open() as f:
        fixed = next(row for row in csv.DictReader(f) if row['case']=='SV100-GT')
    result = [dict(name='fixed', config=fixed['config_path'], checkpoint=fixed['checkpoint_path'],
                   metrics=fixed['evaluation_result'])]
    for directory, suffix in [('poseopt_sv100',''), ('poseopt_sv100_warmup2500','_warmup')]:
        state = read(ROOT/'results'/directory/'state.json')
        for method in ('naive','camp'):
            entry = state['experiments'][method+'/SV100-GT']
            if entry['status'] != 'SUCCESS':
                raise ValueError('Incomplete source run')
            result.append(dict(name=method+suffix, config=entry['config'], checkpoint=entry['checkpoint'],
                               metrics=str(Path(entry['attempt'])/'metrics.json')))
    return result


def worker(args):
    import torch
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.model_components.scene_colliders import NearFarCollider
    job = next(j for j in jobs() if j['name']==args.worker)
    out = args.output/job['name']; out.mkdir(parents=True, exist_ok=True)
    def configure(config):
        config.pipeline.datamanager.camera_res_scale_factor = args.resolution
        config.load_step = 29999
        return config
    torch.manual_seed(42)
    config, pipeline, checkpoint, step = eval_setup(Path(job['config']), test_mode='test', update_config_callback=configure)
    if checkpoint.resolve() != Path(job['checkpoint']).resolve() or step != 29999:
        raise ValueError('Wrong checkpoint')
    pipeline.eval()
    model = pipeline.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dp = copy.deepcopy(config.pipeline.datamanager.dataparser)
    # Ground-truth train and test metadata, never held-out pixel fitting.
    dp.data = GT
    parser = dp.setup()
    gt_train = parser.get_dataparser_outputs(split='train')
    gt_test = parser.get_dataparser_outputs(split='test')
    train = pipeline.datamanager.train_dataset
    if [p.name for p in train.image_filenames] != [p.name for p in gt_train.image_filenames]:
        raise ValueError('Training frame order differs')
    reference = gt_train.cameras.camera_to_worlds.cpu().numpy().astype(np.float64)
    estimated = (model.optimized_poses() if hasattr(model,'optimized_poses') else train.cameras.camera_to_worlds).detach().cpu().numpy()
    sim = fit_sim3(reference[:,:,3], estimated[:,:,3])
    fitted = transform_poses(reference, sim)
    residual = np.linalg.norm(fitted[:,:,3]-estimated[:,:,3], axis=1)
    relative = fitted[:,:,:3] @ estimated[:,:,:3].transpose(0,2,1)
    angles = np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    sim.update(direction='GT parser world -> reconstruction world', fit='training camera centers only',
               center_residual_mean=float(residual.mean()), rotation_residual_deg_mean=float(angles.mean()))
    save(out/'alignment.json', sim)
    loader = pipeline.datamanager.fixed_indices_eval_dataloader
    np.testing.assert_allclose(loader.cameras.camera_to_worlds.cpu().numpy(), gt_test.cameras.camera_to_worlds.numpy(), atol=1e-6)
    if not isinstance(model.collider, NearFarCollider):
        raise ValueError('Only fixed near/far collider is supported')
    near, far = model.collider.near_plane, model.collider.far_plane
    indices = list(range(200)) if args.full else list(range(0,200,10))
    if len(loader.cameras)!=200:
        raise ValueError('Expected 200 test frames')
    historical = read(job['metrics'])
    if Path(historical['checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('Historical metrics checkpoint mismatch')
    per_view=[]
    start=time.time()
    with torch.no_grad():
        for index in indices:
            camera, batch = loader.get_camera(index)
            # Formal raw metrics are reused from the same checkpoint and original evaluator.
            modes = ('aligned',) if args.full else ('raw','aligned')
            for mode in modes:
                current=copy.deepcopy(camera)
                if mode=='aligned':
                    pose=transform_poses(current.camera_to_worlds.cpu().numpy(),sim)
                    current.camera_to_worlds=torch.as_tensor(pose,dtype=current.camera_to_worlds.dtype,device=current.device)
                scale=sim['scale'] if mode=='aligned' else 1.
                model.collider.near_plane=near*scale
                model.collider.far_plane=far*scale
                # Collider inference resets near to zero: zero remains zero under scaling.
                outputs=model.get_outputs_for_camera(current)
                metrics,_=model.get_image_metrics_and_images(outputs,batch)
                row=dict(view=index,mode=mode,**{k:metrics['fine_'+k] for k in ('psnr','ssim','lpips')})
                if not all(np.isfinite(row[k]) for k in ('psnr','ssim','lpips')):
                    raise ValueError('Nonfinite metrics')
                per_view.append(row)
                with (out/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                if index in (0,50,100):
                    rgb=outputs['rgb_fine'].clamp(0,1).cpu().numpy()
                    Image.fromarray((rgb*255).astype('uint8')).save(out/('%s_%03d.png'%(mode,index)))
                    gt=model.renderer_rgb.blend_background(batch['image'].to(model.device)).clamp(0,1).cpu().numpy()
                    Image.fromarray((gt*255).astype('uint8')).save(out/('gt_%03d.png'%index))
            print(job['name'], index, 'elapsed',round(time.time()-start,1),flush=True)
    mean={mode:{key:float(np.mean([r[key] for r in per_view if r['mode']==mode])) for key in ('psnr','ssim','lpips')} for mode in modes}
    if args.full:
        mean['raw']={k:historical['results']['fine_'+k] for k in ('psnr','ssim','lpips')}
    if job['name']=='fixed':
        for key in ('psnr','ssim','lpips'):
            if abs(mean['raw'][key]-mean['aligned'][key]) > (0.01 if key=='psnr' else 1e-4):
                raise ValueError('Identity rendering check failed: '+key)
    save(out/'result.json',dict(job=job,step=step,resolution=args.resolution,indices=indices,alignment=sim,
         metrics=mean,per_view=per_view,elapsed_seconds=time.time()-start,
         raw_source='existing full-resolution 200-view metrics' if args.full else 'paired rerender',
         sampling_bounds=dict(original_near=near,original_far=far,aligned_near=near*sim['scale'],aligned_far=far*sim['scale'],
                              reset_near_plane=model.collider.reset_near_plane)))


def report(output):
    results=[read(output/j['name']/'result.json') for j in jobs() if (output/j['name']/'result.json').exists()]
    rows=[]
    lines=['# GT-condition alignment diagnostic','',
           'Alignment is fitted using GT and optimized TRAIN camera centers only. No test-image fitting, no retraining.',
           'The same transform rotates test orientations and scales/translates test centers. Sampling bounds scale consistently.',
           'A center-fit Sim(3) cannot remove local pose errors; residual degradation does not identify its cause.','']
    for result in results:
        m=result['metrics']; a=result['alignment']
        rows.append(dict(method=result['job']['name'],raw_psnr=m['raw']['psnr'],aligned_psnr=m['aligned']['psnr'],
                         delta_psnr=m['aligned']['psnr']-m['raw']['psnr'],raw_ssim=m['raw']['ssim'],aligned_ssim=m['aligned']['ssim'],
                         raw_lpips=m['raw']['lpips'],aligned_lpips=m['aligned']['lpips'],scale=a['scale'],
                         center_residual=a['center_residual_mean'],rotation_residual_deg=a['rotation_residual_deg_mean']))
    if not rows: return
    lines += ['Resolution scale: %s; test views: %s. Single scene and training seed.'%(results[0]['resolution'],len(results[0]['indices'])),
              'Raw metrics source: '+results[0]['raw_source']+'.','',
              '| Method | Raw PSNR | Aligned PSNR | Delta | Raw SSIM | Aligned SSIM | Raw LPIPS | Aligned LPIPS | Scale |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {method} | {raw_psnr:.3f} | {aligned_psnr:.3f} | {delta_psnr:+.3f} | {raw_ssim:.4f} | {aligned_ssim:.4f} | {raw_lpips:.4f} | {aligned_lpips:.4f} | {scale:.5f} |'.format(**r))
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    with (output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if len(results)==5 and results[0]['resolution']<1:
        from PIL import Image, ImageDraw
        for view in (0,50,100):
            canvas=Image.new('RGB',(6*200,2*230),'white'); draw=ImageDraw.Draw(canvas)
            gt=Image.open(output/'fixed'/('gt_%03d.png'%view))
            for row, mode in enumerate(('raw','aligned')):
                canvas.paste(gt,(0,row*230+30));draw.text((5,row*230+5),'GT '+mode,fill='black')
                for col,j in enumerate(jobs(),1):
                    canvas.paste(Image.open(output/j['name']/('%s_%03d.png'%(mode,view))),(col*200,row*230+30))
                    draw.text((col*200+5,row*230+5),j['name'],fill='black')
            canvas.save(output/('comparison_%03d.png'%view))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--worker',choices=[j['name'] for j in jobs()])
    args=parser.parse_args();args.output=args.output.resolve();args.resolution=1. if args.full else .25
    if args.worker: return worker(args)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(full=args.full,resolution=args.resolution,jobs=jobs(),
                  code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'poseguard/alignment.py']})
    path=args.output/'manifest.json'
    if path.exists() and read(path)!=manifest: raise ValueError('Run manifest mismatch')
    save(path,manifest)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env['NERFSTUDIO_METHOD_CONFIGS']='poseguard-naive=poseguard.methods:naive,poseguard-camp=poseguard.methods:camp'
    for j in jobs():
        if (args.output/j['name']/'result.json').exists(): continue
        save(args.output/'state.json',dict(status='RUNNING',active=j['name']))
        cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(args.output),'--worker',j['name']]
        if args.full:cmd.append('--full')
        with (args.output/(j['name']+'.log')).open('a') as log:
            rc=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if rc:
            save(args.output/'state.json',dict(status='FAILED',active=j['name'],returncode=rc))
            raise RuntimeError('Worker failed: '+j['name'])
        report(args.output)
    report(args.output)
    save(args.output/'state.json',dict(status='SUCCESS'))

if __name__=='__main__':main()
