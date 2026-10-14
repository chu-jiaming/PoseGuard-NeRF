"""Inference only. Uses installed training environment and original transform_poses."""
import os,sys,json,time,copy,argparse,subprocess
from pathlib import Path
ROOT=Path('/root/workspace/poseguard-nerf'); OUT=ROOT/'delivery_20261014'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(OUT/'runtime_cache/matplotlib')
os.environ['NERFSTUDIO_METHOD_CONFIGS']='poseguard-naive=poseguard.methods:naive,poseguard-camp=poseguard.methods:camp'
sys.path.insert(0,str(ROOT));os.chdir(str(ROOT))
import numpy as np
from resume_io import png, valid, records, save_record, lock

def write_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def append(row):
 with (OUT/'render_records.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def safety():
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 if gpu:raise RuntimeError('GPU has running compute processes: '+gpu)
 p=ROOT/'.sv100_pipeline.lock';s=p.stat();key='%02x:%02x:%d'%(os.major(s.st_dev),os.minor(s.st_dev),s.st_ino)
 if key in Path('/proc/locks').read_text():raise RuntimeError('Pipeline lock occupied')
 processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
 for line in processes.splitlines():
  if any(x in line for x in ['ns-train ', 'ns-eval ', 'poseguard.worker train', 'poseguard.worker evaluate']):
   raise RuntimeError('Training/evaluation process: '+line)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--condition',required=True);ap.add_argument('--method',required=True);ap.add_argument('--mode',choices=['raw','aligned'],default='raw');ap.add_argument('--video',action='store_true');ap.add_argument('--benchmark',action='store_true');a=ap.parse_args()
 lock_fd=lock();safety();start=time.time()
 import torch
 from PIL import Image
 import matplotlib.cm as cm
 from nerfstudio.utils.eval_utils import eval_setup
 from nerfstudio.model_components.scene_colliders import NearFarCollider
 from poseguard.alignment import transform_poses
 inventory=json.loads((OUT/'inventory.json').read_text());assert inventory['render_gate_passed']
 job=next(r for r in inventory['runs'] if r['condition']==a.condition and r['method']==a.method)
 assert job['verification_passed']
 def configure(c):
  c.load_step=29999
  c.pipeline.datamanager.camera_res_scale_factor=.5 if a.video else 1.
  return c
 torch.manual_seed(42)
 config,pipeline,checkpoint,step=eval_setup(Path(job['config']),test_mode='test',update_config_callback=configure)
 assert checkpoint.resolve()==Path(job['checkpoint']).resolve() and step==29999
 pipeline.eval();model=pipeline.model
 for p in model.parameters():p.requires_grad_(False)
 assert isinstance(model.collider,NearFarCollider)
 assert model.collider.reset_near_plane and model.collider.near_plane==2 and model.collider.far_plane==6
 assert model.config.background_color=='white'
 loader=pipeline.datamanager.fixed_indices_eval_dataloader
 assert len(loader.cameras)==200
 dp=copy.deepcopy(config.pipeline.datamanager.dataparser);dp.data=Path('/root/workspace/data/nerf_synthetic/lego')
 gt=dp.setup().get_dataparser_outputs(split='test');gt.cameras.rescale_output_resolution(.5 if a.video else 1.)
 np.testing.assert_allclose(loader.cameras.camera_to_worlds.cpu().numpy(),gt.cameras.camera_to_worlds.numpy(),atol=1e-6)
 for key in ['fx','fy','cx','cy','width','height']:
  np.testing.assert_allclose(getattr(loader.cameras,key).cpu().numpy(),getattr(gt.cameras,key).numpy(),atol=1e-6)
 assert [p.name for p in loader.input_dataset.image_filenames]==[p.name for p in gt.image_filenames]
 sim=None;reference={};refpath=None
 if a.mode=='aligned':
  assert a.condition=='SV100-GT' and a.method!='fixed'
  sp=ROOT/'delivery_ref/alignment_full'/a.method/'SV100-GT/alignment.json';sim=json.loads(sp.read_text())
  original=json.loads((ROOT/'results/alignment_gt_full'/a.method/'alignment.json').read_text())
  for key in ['scale','rotation','translation']:np.testing.assert_allclose(sim[key],original[key],rtol=0,atol=1e-12)
  if a.method=='naive':assert abs(sim['scale']-1.0108)<.0001
  near,far=model.collider.near_plane,model.collider.far_plane
  model.collider.near_plane=near*sim['scale'];model.collider.far_plane=far*sim['scale']
  refpath=ROOT/'results/alignment_gt_full'/a.method/'progress.jsonl'
 elif a.condition=='SV100-GT' and a.method=='fixed':
  # Historical aligned fixed uses exactly identity Sim(3), hence is a raw reference.
  ident=json.loads((ROOT/'results/alignment_gt_full/fixed/alignment.json').read_text())
  assert abs(ident['scale']-1)<1e-12;np.testing.assert_allclose(ident['rotation'],np.eye(3),atol=1e-12);np.testing.assert_allclose(ident['translation'],np.zeros(3),atol=1e-12)
  refpath=ROOT/'results/alignment_gt_full/fixed/progress.jsonl'
 if refpath:
  result=json.loads((refpath.parent/'result.json').read_text());assert result['resolution']==1 and len(result['indices'])==200
  for row in map(json.loads,refpath.read_text().splitlines()):
   if row['mode']=='aligned':reference[row['view']]=row
  assert len(reference)==200
 indices=list(range(200)) if a.video else ([0] if a.benchmark else [0,50,100,175])
 previous=records()
 command='PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python '+str(Path(__file__))+' '+' '.join(sys.argv[1:])
 rows=[]
 for idx in indices:
  matching=[r for r in previous if r['condition']==a.condition and r['method']==a.method and r['mode']==a.mode and r['view']==idx and r['video']==a.video]
  target_check=OUT/'videos/frames'/('%s_%s_%s'%(a.condition,a.method,a.mode))/('view_%03d.png'%idx) if a.video else OUT/'stills'/a.condition/a.method/('view_%03d_%s.png'%(idx,a.mode))
  image_ok=valid(target_check,(400,400) if a.video else (800,800))
  if image_ok and matching:
   assert len(matching)==1;rows.append(matching[0]);continue
  if target_check.exists() and not image_ok:target_check.unlink()
  camera,batch=loader.get_camera(idx);current=copy.deepcopy(camera)
  if sim:
   pose=transform_poses(current.camera_to_worlds.cpu().numpy(),sim)
   current.camera_to_worlds=torch.as_tensor(pose,dtype=current.camera_to_worlds.dtype,device=current.device)
  torch.cuda.synchronize();tic=time.time()
  with torch.no_grad():
   outputs=model.get_outputs_for_camera(current)
   torch.cuda.synchronize();render_seconds=time.time()-tic
   metrics,_=model.get_image_metrics_and_images(outputs,batch)
   metrics={k:float(metrics['fine_'+k]) for k in ['psnr','ssim','lpips']}
   pred=outputs['rgb_fine'].clamp(0,1).cpu().numpy()
   truth=model.renderer_rgb.blend_background(batch['image'].to(model.device)).clamp(0,1).cpu().numpy()
  assert all(np.isfinite(list(metrics.values())))
  assert np.isfinite(pred).all() and np.isfinite(truth).all()
  check=None
  if not a.video and idx in reference:
   delta=abs(metrics['psnr']-reference[idx]['psnr'])
   check=dict(type='per_view',source=str(refpath),reference_PSNR=reference[idx]['psnr'],delta_dB=delta,tolerance_dB=.05,passed=delta<=.05)
   if delta>.05:
    write_json(OUT/'STOP.json',dict(condition=a.condition,method=a.method,mode=a.mode,view=idx,metrics=metrics,check=check));raise RuntimeError('PSNR self-check failed: '+str(check))
  if not a.video and a.mode=='aligned':
   raw=next(r for r in previous if r['condition']==a.condition and r['method']==a.method and r['mode']=='raw' and r['view']==idx and not r['video'])
   if metrics['psnr']<raw['metrics']['psnr']:
    write_json(OUT/'STOP.json',dict(reason='aligned below raw',condition=a.condition,method=a.method,view=idx,aligned=metrics,raw=raw['metrics']));raise RuntimeError('Aligned below raw')
  if a.video:
   target=OUT/'videos/frames'/('%s_%s_%s'%(a.condition,a.method,a.mode))/('view_%03d.png'%idx);err=None;gtpath=None
  else:
   target=OUT/'stills'/a.condition/a.method/('view_%03d_%s.png'%(idx,a.mode));err=target.with_name('view_%03d_err_%s.png'%(idx,a.mode));gtpath=OUT/'stills'/a.condition/'gt'/('view_%03d.png'%idx)
  target.parent.mkdir(parents=True,exist_ok=True)
  if err:
   error=np.linalg.norm(pred-truth,axis=-1);heat=cm.get_cmap('turbo')(np.clip(error/.3,0,1))[...,:3];png(Image.fromarray((heat*255).astype('uint8')),err)
   gtpath.parent.mkdir(parents=True,exist_ok=True)
   if not gtpath.exists():png(Image.fromarray((truth*255).astype('uint8')),gtpath)
  row=dict(condition=a.condition,method=a.method,mode=a.mode,view=idx,video=a.video,config=job['config'],checkpoint=str(checkpoint.resolve()),metrics=metrics,near_effective=0.,far_effective=float(model.collider.far_plane),reset_near_plane=bool(model.collider.reset_near_plane),near_configured=float(model.collider.near_plane),width=pred.shape[1],height=pred.shape[0],file=str(target.relative_to(OUT)),error_file=str(err.relative_to(OUT)) if err else None,gt_file=str(gtpath.relative_to(OUT)) if gtpath else None,gt_source=str(gt.image_filenames[idx]),alignment_source=str(sp) if sim else None,render_seconds=render_seconds,check=check,generation_command=command)
  row['float_rgb_finite_checked']=True
  save_record(row)
  if not image_ok:png(Image.fromarray((pred*255).astype('uint8')),target)
  rows.append(row)
  if a.video:write_json(OUT/'gates/last_frame_written.json',dict(file=str(target),time=time.time(),view=idx))
  print(json.dumps(dict(condition=a.condition,method=a.method,mode=a.mode,view=idx,video=a.video,PSNR=metrics['psnr'],render_seconds=round(render_seconds,3),check=check)),flush=True)
  if a.benchmark:
   # P0:36, P1:60 (including optional T1), P2:800 quarter-size frames.
   estimate=render_seconds*(36+60+800*.25)*1.25
   write_json(OUT/'timing_estimate.json',dict(single_800_render_seconds=render_seconds,still_count=96,video_frames=800,quarter_resolution_cost_assumption=.25,overhead_factor=1.25,estimated_total_seconds=estimate,exceeds_8_hours=estimate>8*3600,include_T1=True))
 if not a.video and not a.benchmark and a.mode=='raw':
  selected=[r['metrics']['psnr'] for r in rows if r['view'] in [0,50,100]];avg=float(np.mean(selected));historical=json.loads(Path(job['metrics'][0]).read_text())['results'];tol=max(3.,3*historical['fine_psnr_std']);delta=abs(avg-job['PSNR'])
  summary=dict(condition=a.condition,method=a.method,mode='raw',views=[0,50,100],mean_psnr=avg,full_200_PSNR=job['PSNR'],delta_dB=delta,same_magnitude_criterion_dB=tol,passed=delta<=tol,per_view_reference_available=bool(reference))
  write_json(OUT/'checks'/('%s_%s_raw.json'%(a.condition,a.method)),summary)
  if not summary['passed']:write_json(OUT/'STOP.json',summary);raise RuntimeError('Raw three-view mean far from historical full mean')
 print('DONE',a.condition,a.method,a.mode,'elapsed',round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
