#!/usr/bin/env python
"""Fresh SV100 orchestration. Definitions come from existing sweep scripts/metadata."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/root/workspace/poseguard-nerf')
DATA = Path('/root/workspace/data/nerf_synthetic')
GT = DATA / 'lego'
CASES = [('SV100-GT', 'gt', 0.0, 'lego')]
CASES += [('SV100-T'+label, 'translation', value, dataset) for label, value, dataset in [
    ('0.5', .005, 'lego_sv100_tnoise_0p5pct'), ('1', .01, 'lego_tnoise_1pct'),
    ('2', .02, 'lego_sv100_tnoise_2pct'), ('5', .05, 'lego_sv100_tnoise_5pct')]]
CASES += [('SV100-R'+label, 'rotation', value, 'lego_sv100_rnoise_'+suffix+'deg')
          for label, value, suffix in [('0.5', .5, '0p5'), ('1', 1., '1'), ('2', 2., '2'), ('5', 5., '5')]]
FIELDS = ['case', 'perturbation_type', 'perturbation_magnitude', 'dataset_path', 'training_views',
          'nerfstudio_version', 'max_iterations', 'output_path', 'config_path', 'checkpoint_path',
          'evaluation_result', 'psnr', 'ssim', 'lpips', 'status']
BAD = re.compile(r'Traceback \(most recent call last\)|CUDA.*out of memory|CUDA OOM', re.I)

def now():
    return datetime.now(timezone.utc).isoformat()

def log(s):
    print(now(), s, flush=True)

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def version():
    v = importlib.metadata.version('nerfstudio')
    require(v == '1.1.5', 'Wrong Nerfstudio version: '+v)
    return v

def read(p):
    return json.loads(p.read_text())

def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n')

def validate(case):
    import numpy as np
    name, kind, mag, directory = case
    version()
    d = DATA / directory
    require(d.is_dir(), 'Dataset missing: '+str(d))
    fingerprints = {}
    for split in ['train', 'val', 'test']:
        p = d / ('transforms_'+split+'.json')
        obj = read(p)
        fingerprints[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        require(len(obj['frames']) == (100 if split == 'train' else len(read(GT/p.name)['frames'])), 'Wrong split count: '+str(p))
        for f in obj['frames']:
            image = d / (f['file_path'].replace('./', '')+'.png')
            require(image.is_file(), 'Missing image: '+str(image))
            matrix = np.asarray(f['transform_matrix'])
            require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), 'Invalid camera: '+str(p))
        if kind != 'gt' and split != 'train':
            require(obj == read(GT/p.name), 'Changed evaluation split: '+str(p))
    if kind != 'gt':
        m = read(d/'pose_noise_metadata.json')
        require(Path(m['source_dataset']).resolve() == GT and m['num_training_cameras'] == 100 and m['seed'] == 42, 'Metadata source/count/seed mismatch: '+name)
        require(m['noise_type'] == kind+'_only', 'Noise type mismatch: '+name)
        require(m['translation_ratio'] == (mag if kind == 'translation' else 0), 'Translation mismatch: '+name)
        require(m.get('rotation_deg', 0) == (mag if kind == 'rotation' else 0), 'Rotation mismatch: '+name)
        require(read(d/'transforms_train_gt.json') == read(GT/'transforms_train.json'), 'GT reference mismatch: '+name)
        a, b = read(d/'transforms_train.json'), read(GT/'transforms_train.json')
        require(a['camera_angle_x'] == b['camera_angle_x'], 'Intrinsics mismatch')
        for x, y in zip(a['frames'], b['frames']):
            require(x['file_path'] == y['file_path'], 'Training split changed')
            x, y = np.array(x['transform_matrix']), np.array(y['transform_matrix'])
            if kind == 'translation':
                require(np.array_equal(x[:3,:3], y[:3,:3]), 'Translation changed rotation')
                require(np.isclose(np.linalg.norm(x[:3,3]-y[:3,3]), m['translation_magnitude']), 'Translation magnitude mismatch')
            else:
                require(np.array_equal(x[:3,3], y[:3,3]), 'Rotation changed translation')
                delta = x[:3,:3] @ np.linalg.inv(y[:3,:3])
                angle = np.rad2deg(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
                require(np.isclose(angle, mag, atol=1e-5), 'Rotation magnitude mismatch')
        fingerprints['pose_noise_metadata.json'] = hashlib.sha256((d/'pose_noise_metadata.json').read_bytes()).hexdigest()
    log('DATA OK '+name+' dataset='+str(d)+' training_views=100 nerfstudio_version=1.1.5')
    return fingerprints

def command(cmd, path):
    log('COMMAND '+shlex.join([str(x) for x in cmd])+' LOG '+str(path))
    with path.open('x') as out:
        out.write(now()+' START '+shlex.join([str(x) for x in cmd])+'\n'); out.flush()
        p = subprocess.Popen([str(x) for x in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout:
                out.write(line); out.flush()
                print(line, end='', flush=True)
                if BAD.search(line):
                    p.terminate()
                    raise RuntimeError('Fatal error in '+str(path))
            code = p.wait()
            require(code == 0, 'Nonzero exit '+str(code)+': '+str(path))
            out.write(now()+' SUCCESS END\n')
        finally:
            if p.poll() is None:
                p.kill(); p.wait()

def verify_config(config, dataset):
    import yaml
    import torch
    c = yaml.load(config.read_text(), Loader=yaml.Loader)
    require(c.method_name == 'vanilla-nerf' and c.max_num_iterations == 30000, 'Config model/iterations mismatch')
    require(c.load_dir is None and c.load_checkpoint is None, 'Not a fresh training config')
    # Nerfstudio saves config before DataManager applies the --data alias.
    dm = c.pipeline.datamanager
    effective_data = dm.data if dm.data is not None else dm.dataparser.data
    require(Path(effective_data).resolve() == dataset.resolve(), 'Config dataset mismatch')
    dm.dataparser.data = Path(effective_data)
    from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
    require(isinstance(c.pipeline.datamanager.dataparser, BlenderDataParserConfig), 'Wrong parser')
    parsed = c.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split='train')
    require(len(parsed.image_filenames) == 100, 'Parser training count mismatch')
    ckpt = c.get_checkpoint_dir().resolve() / 'step-000029999.ckpt'
    require(ckpt.is_file(), '30k checkpoint missing')
    state = torch.load(ckpt, map_location='cpu')
    require(state['step'] == 29999 and bool(state['pipeline']), 'Checkpoint incomplete')
    log('VALIDATED config='+str(config)+' checkpoint='+str(ckpt)+' iterations=30000')
    return ckpt

def sanity(config, target):
    import torch
    import numpy as np
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup
    _, pipeline, checkpoint, step = eval_setup(config, test_mode='test')
    require(step == 29999, 'Sanity checkpoint step mismatch')
    camera, _ = next(iter(pipeline.datamanager.fixed_indices_eval_dataloader))
    with torch.no_grad():
        rgb = pipeline.model.get_outputs_for_camera(camera)['rgb_fine'].detach().cpu().numpy()
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and np.isfinite(rgb).all(), 'Invalid render tensor')
    Image.fromarray((np.clip(rgb, 0, 1)*255).astype('uint8')).save(target)
    with Image.open(target) as im:
        im.load()
        require(im.width > 0 and im.height > 0 and np.asarray(im).std() > 0, 'Invalid render image')
    log('RENDER SANITY SUCCESS '+str(target)+' checkpoint='+str(checkpoint))

def summaries(rows, status):
    for name, subset in [('translation_sweep', [r for r in rows if r['perturbation_type'] in ['gt','translation']]),
                         ('rotation_sweep', [r for r in rows if r['perturbation_type'] in ['gt','rotation']]), ('full_summary', rows)]:
        p = ROOT/'results'/('SV100_'+name+'.csv')
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(subset)
    (ROOT/'results/SV100_pipeline_summary.txt').write_text(now()+' '+status+'\nNerfstudio 1.1.5; vanilla-nerf; 100 views; 30000 iterations\nTranslation magnitude unit: ratio of mean camera radius; rotation unit: degrees.\nGT is evaluated once and shared by both sweep tables. Metrics: fine_psnr/fine_ssim/fine_lpips from fresh ns-eval test split.\n'+ '\n'.join(r['case']+' '+r['status']+' PSNR='+str(r['psnr'])+' SSIM='+str(r['ssim'])+' LPIPS='+str(r['lpips']) for r in rows)+'\n')

def main():
    rows = []
    state = {'status':'RUNNING','start':now(),'pid':os.getpid(),'nerfstudio_version':version(),'order':[c[0] for c in CASES]}
    state_path = ROOT/'results/pipeline_state.json'
    require(not state_path.exists(), 'Existing pipeline state: refusing result reuse')
    save(state_path, state)
    try:
        checks = {c[0]:validate(c) for c in CASES}
        save(ROOT/'results/dataset_manifest.json', checks)
        subprocess.run([sys.executable,'-m','pip','freeze'], stdout=(ROOT/'results/environment.txt').open('w'), check=True)
        summaries(rows, 'RUNNING')
        for case in CASES:
            name, kind, mag, directory = case
            state.update(case=name, stage='train'); save(state_path,state)
            require(validate(case) == checks[name], 'Dataset metadata changed after preflight')
            log('START '+name+' max_iterations=30000')
            logs = ROOT/'logs'/kind; logs.mkdir(exist_ok=True)
            result = ROOT/'results'/name; result.mkdir()
            exp = 'lego_sv100' if kind == 'gt' else directory
            base = ROOT/'outputs'/exp
            require(not base.exists(), 'Output experiment already exists; refusing reuse: '+str(base))
            before = set((ROOT/'outputs').rglob('config.yml'))
            command(['ns-train','vanilla-nerf','--experiment-name',exp,'--output-dir','outputs','--max-num-iterations','30000','--vis','tensorboard','--viewer.quit-on-train-completion','True','--data',DATA/directory], logs/(name+'_train.log'))
            configs = set((ROOT/'outputs').rglob('config.yml'))-before
            require(len(configs) == 1, 'Fresh config not uniquely identified')
            config = configs.pop().resolve()
            require(base.resolve() in config.parents, 'Unexpected config location')
            state.update(stage='validate',config_path=str(config)); save(state_path,state)
            command([sys.executable,__file__,'verify',config,DATA/directory], logs/(name+'_validate.log'))
            ckpt = config.parent/'nerfstudio_models/step-000029999.ckpt'
            require(ckpt.is_file(), 'Checkpoint missing')
            state['stage']='eval'; save(state_path,state)
            metrics = result/'metrics.json'
            command(['ns-eval','--load-config',config,'--output-path',metrics], logs/(name+'_eval.log'))
            raw = read(metrics)
            require(Path(raw['checkpoint']).resolve() == ckpt.resolve(), 'Evaluation used wrong checkpoint')
            vals = {k:float(raw['results']['fine_'+k]) for k in ['psnr','ssim','lpips']}
            require(all(math.isfinite(x) for x in vals.values()), 'Nonfinite metrics')
            log(name+' METRICS '+json.dumps(vals))
            if kind == 'gt':
                state['stage']='render_sanity'; save(state_path,state)
                command([sys.executable,__file__,'sanity',config,result/'test_sanity.png'], logs/(name+'_render.log'))
            row = dict(case=name, perturbation_type=kind, perturbation_magnitude=mag, dataset_path=str(DATA/directory),training_views=100,nerfstudio_version='1.1.5',max_iterations=30000,output_path=str(config.parent),config_path=str(config),checkpoint_path=str(ckpt),evaluation_result=str(metrics),status='SUCCESS',**vals)
            save(result/'metadata.json',row); rows.append(row); summaries(rows,'RUNNING')
            log('SUCCESS END '+name)
        summaries(rows,'SUCCESS'); state.update(status='SUCCESS',stage='complete',end=now()); save(state_path,state)
        log('SUCCESS END FULL PIPELINE')
    except BaseException as e:
        state.update(status='FAILED',error=str(e),end=now()); save(state_path,state)
        summaries(rows,'FAILED '+str(e)); log('FAILED END '+str(e)); raise

if __name__ == '__main__':
    os.chdir(ROOT)
    if len(sys.argv) == 1: main()
    elif sys.argv[1] == 'preflight':
        for c in CASES: validate(c)
    elif sys.argv[1] == 'verify': verify_config(Path(sys.argv[2]),Path(sys.argv[3]))
    elif sys.argv[1] == 'sanity': sanity(Path(sys.argv[2]),Path(sys.argv[3]))
    else: raise ValueError('Unknown mode')
