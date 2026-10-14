"""Independent delivery gates. Never invokes the next gate."""
import os,sys,json,time,subprocess,shutil,hashlib,tarfile,csv,traceback
from pathlib import Path
from fractions import Fraction
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from resume_io import atomic_json,png,valid,records,lock
D=Path(__file__).resolve().parents[1];G=D/'gates';G.mkdir(exist_ok=True)
SEQS=[('SV100-GT','fixed','raw'),('SV100-GT','naive_warmup','aligned'),('SV100-T5','fixed','raw'),('SV100-T5','naive_warmup','raw')]
checks=[];details={};fd=None
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists() else ImageFont.load_default()
def check(name,ok,detail=None):
 checks.append(dict(name=name,passed=bool(ok),detail=detail))
 if not ok:raise RuntimeError(name+': '+str(detail))
def run(script,args=[]):
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['POSEGUARD_LOCK_FD']=str(fd)
 subprocess.run([sys.executable,str(D/'scripts'/script)]+args,env=env,pass_fds=(fd,),check=True)
def prereq(g):check('Gate '+g+' PASSED',(G/(g+'.json')).exists() and json.loads((G/(g+'.json')).read_text())['status']=='PASSED')
def audit_frames():
 rs={r['file']:r for r in records()};video=[];sizes=0
 for c,m,mode in SEQS:
  name='_'.join((c,m,mode));folder=D/'videos/frames'/name;ps=sorted(folder.glob('*.png'))
  check(name+' count and indices',set(p.name for p in ps)=={'view_%03d.png'%i for i in range(200)},len(ps))
  scale=json.loads((D.parent/'delivery_ref/alignment_full'/m/'SV100-GT/alignment.json').read_text())['scale'] if mode=='aligned' else 1
  for i in range(200):
   p=folder/('view_%03d.png'%i);check(name+' frame '+str(i)+' readable/size',valid(p,(400,400)))
   arr=np.asarray(Image.open(p));check(name+' frame '+str(i)+' content',np.isfinite(arr).all() and not np.all(arr==255) and not np.all(arr==0))
   r=rs.get(str(p.relative_to(D)));check(name+' frame '+str(i)+' metrics',r is not None and r['video'] and r['width']==400 and r['height']==400 and all(np.isfinite(r['metrics'][k]) for k in ['psnr','ssim','lpips']))
   check(name+' frame '+str(i)+' effective bounds',r['near_effective']==0 and abs(r['far_effective']-6*scale)<1e-10 and r['reset_near_plane'])
   video.append(r);sizes+=p.stat().st_size
 atomic_json(G/'A_metrics_400.json',dict(resolution=[400,400],comparison_with_800_metrics=False,note='400x400 metrics are independent; not compared against 800x800 references. PNG cannot encode floating NaN; decoded pixels and metric finiteness checked.',frames=video))
 return sizes

def A():
 old=records();lookup={r['file']:r for r in old}
 for c,m,mode in SEQS:
  folder=D/'videos/frames'/('_'.join((c,m,mode)))
  if all(valid(folder/('view_%03d.png'%i),(400,400)) and str((folder/('view_%03d.png'%i)).relative_to(D)) in lookup for i in range(200)):
   print('REUSE COMPLETE',folder.name,flush=True);continue
  run('render.py',['--condition',c,'--method',m,'--mode',mode,'--video'])
 details['frame_bytes']=audit_frames()
 canvas=Image.new('RGB',(1600,920),'white');draw=ImageDraw.Draw(canvas)
 for row,(c,m,mode) in enumerate(SEQS):
  name='_'.join((c,m,mode));draw.text((5,row*230+2),name,font=FONT,fill='black')
  for col,i in enumerate(range(0,200,25)):
   with Image.open(D/'videos/frames'/name/('view_%03d.png'%i)) as im:canvas.paste(im.resize((200,200)),(col*200,row*230+30))
   draw.text((col*200+4,row*230+16),str(i),fill='black')
 png(canvas,G/'A_contact_sheet.png');check('contact sheet <=1600px',canvas.width<=1600)

def B():
 prereq('A');probe=shutil.which('ffprobe');check('ffprobe available',probe is not None,'Required ffprobe not found; no installation or substitute attempted.')
 details['frame_bytes']=audit_frames();details['exclude_frames_from_archive']=details['frame_bytes']>300_000_000
 run('encode_videos.py')
 names=['_'.join(s) for s in SEQS]+['side_by_side_T5']
 for name in names:
  p=D/'videos'/(name+'.mp4');j=json.loads(subprocess.check_output([probe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,avg_frame_rate,nb_read_frames,duration:format=duration','-of','json',str(p)],text=True));s=j['streams'][0];duration=float(s.get('duration',j['format']['duration']))
  check(name+' ffprobe',s['codec_name']=='h264' and s['pix_fmt']=='yuv420p' and Fraction(s['avg_frame_rate'])==30 and int(s['nb_read_frames'])==200 and abs(duration-200/30)<.05,j)
  if name!='side_by_side_T5':
   p=D/'videos'/(name+'.gif')
   with Image.open(p) as im:check(name+' GIF',im.width<=400 and p.stat().st_size<8_000_000,dict(width=im.width,bytes=p.stat().st_size))
 import av
 with av.open(str(D/'videos/side_by_side_T5.mp4')) as container:
  for idx,frame in enumerate(container.decode(video=0)):
   if idx in [0,100]:
    im=frame.to_image();png(im,G/('B_preview_%03d.png'%idx));arr=np.asarray(im)
    check('top labels frame '+str(idx),all(np.count_nonzero(np.all(arr[:32,x:x+400,:]>180,axis=-1))>30 for x in [0,400]),'Both label regions contain bright text on dark top strip; preview retained for human inspection.')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def C():
 prereq('A');prereq('B');run('export_curves.py')
 inv=json.loads((D/'inventory.json').read_text());runs=inv['runs'];found=sum(bool(r.get('checkpoint')) for r in runs);summary=json.loads((D/'curves/export_summary.json').read_text());tags={}
 for r in summary:
  for tag,n in r['tags'].items():tags[tag]=tags.get(tag,0)+n
 check('all found runs exported',all(r['rows']>0 for r in summary));details.update(found_runs=found,missing_runs=len(runs)-found,tag_rows=tags)
 rs=records();rawchecks=[json.loads(p.read_text()) for p in (D/'checks').glob('*.json')];delta=max(r['delta_dB'] for r in rawchecks)
 exclusion=json.loads((G/'B.json').read_text())['details']['exclude_frames_from_archive'];details['exclude_frames_from_archive']=exclusion
 lines=['# PoseGuard-NeRF delivery','',f'找到 {found} 个run；缺失 {len(runs)-found} 个。核心15个PSNR及两个参考checkpoint哈希已通过。','P0/P1：96张800×800渲染、96张误差图、16张GT；视角0、50、100、175（175展示斜侧机械细节）。','P2：4段200帧400×400视频、4个GIF和带标签的T5并排MP4。P3标量与36份diagnostics；P4见checkpoints_index.json。','raw均值检查最大偏差为 %.1f dB（精确 %.9f dB）。raw检查主要只做均值级比对，没有原始raw全分辨率逐图记录可供核对；例外是GT/fixed另用历史恒等对齐逐图指标核对过4张图，偏差0。不能把均值检查当成0.05 dB逐图验证。'%(delta,delta),'四种aligned方法的16张800×800图严格核对历史逐图指标，最大偏差0 dB；400×400指标单独记录，不与800×800比较。','raw实际near=0、far=6；aligned调用原transform_poses，实际near=0、far=6×scale，reset_near_plane=true。白背景，rgb_fine。误差图为未量化RGB每像素L2，turbo [0,0.3]。','目录差异：outputs及poseopt_sv100、部分warmup attempt通过软链接指向/root/autodl-tmp；fixed/T1为lego_tnoise_1pct；aligned progress实际在results/alignment_gt_full。','原代码、实验输出、checkpoint、数据及event文件未修改；没有新训练。','逐帧PNG超过300MB，已从压缩包排除但保留服务器原文件。' if exclusion else '逐帧PNG未超过300MB，已纳入压缩包。','各关耗时和验收见gates/A.json、B.json及C.json；关卡不会自动衔接。','包路径、大小、SHA256见包外gates/C.md及archive_receipt.json；哈希无法自包含于其所校验的压缩包。','','## CSV标签行数','','| tag | rows |','|---|---:|']+[f'| {k} | {n} |' for k,n in sorted(tags.items())]
 lines+=['','## P0/P1 PSNR','','| condition | method | mode | view | PSNR |','|---|---|---|---:|---:|']+[f"| {r['condition']} | {r['method']} | {r['mode']} | {r['view']} | {r['metrics']['psnr']:.6f} |" for r in rs if not r['video']]
 (D/'README.md').write_text('\n'.join(lines)+'\n')
 files={}
 for r in rs:
  for key,kind in [('file','render'),('error_file','error_map'),('gt_file','ground_truth')]:
   if r.get(key):files[r[key]]=dict(r,kind=kind)
 for r in summary:
  for x in r['diagnostics']:files[x['output']]=dict(x,kind='diagnostics',condition=r['condition'],method=r['method'],checkpoint=r['checkpoint'])
 for r in json.loads((D/'videos/encoding_summary.json').read_text()):files[r['file']]=r
 files['curves/tb_scalars.csv']=dict(sources=summary,tag_rows=tags,generation_command='scripts/run_gate.sh C')
 excluded=['runtime_cache/','logs/','gates/C.json','gates/C.md','archive_receipt.json','*.tar.gz','manifest.json','.render.lock','completion.log','finalize.log']
 def include(rel):return not any(rel==x or (x.endswith('/') and rel.startswith(x)) for x in excluded) and not rel.endswith(('.tmp','.tar.gz')) and not (exclusion and rel.startswith('videos/frames/'))
 for p in D.rglob('*'):
  rel=str(p.relative_to(D))
  if p.is_file() and include(rel):files.setdefault(rel,dict(kind='supporting_file',generation_command='scripts/run_gate.sh C'));files[rel].update(bytes=p.stat().st_size,sha256=sha(p),included_in_archive=True)
 for rel,v in files.items():v['included_in_archive']=include(rel)
 atomic_json(D/'manifest.json',dict(files=files,error_map=dict(range=[0,.3],colormap='turbo'),excluded=excluded,raw_mean_max_delta_dB=delta))
 archive=D.parent/'delivery_20261014_final.tar.gz';tmp=D/'delivery_20261014_final.tar.gz.tmp'
 tar_args=['tar','-czf',str(tmp),'--exclude=*.tar.gz','--exclude=*.tmp']
 for rel in excluded:
  if rel not in ['manifest.json','*.tar.gz']:tar_args.append('--exclude='+D.name+'/'+rel.rstrip('/'))
 if exclusion:tar_args.append('--exclude='+D.name+'/videos/frames')
 # Create the excluded output before tar snapshots the source directory metadata.
 tmp.touch(exist_ok=True)
 subprocess.run(tar_args+['--',D.name+'/'],cwd=D.parent,check=True)
 with tarfile.open(tmp,'r:gz') as t:
  members=t.getmembers();file_count=sum(m.isfile() for m in members)
  check('archive readable, no checkpoints/events',all(not m.name.endswith('.ckpt') and 'tfevents' not in m.name for m in members))
  check('archive contains no tar.gz',all(not m.name.endswith('.tar.gz') for m in members))
 os.replace(tmp,archive)
 receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),file_count=file_count);atomic_json(D/'archive_receipt.json',receipt);details['archive']=receipt

def main():
 global fd
 gate=sys.argv[1] if len(sys.argv)==2 else '';assert gate in ['A','B','C'],'Usage: run_gate.sh A|B|C'
 start=time.time();status='FAILED';error=None
 try:
  fd=lock();os.set_inheritable(fd,True)
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();check('GPU idle at startup',not gpu,gpu or 'No compute process')
  globals()[gate]();status='PASSED'
 except BaseException as e:error=str(e);traceback.print_exc()
 result=dict(gate=gate,status=status,checks=checks,details=details,error=error,elapsed_seconds=time.time()-start)
 atomic_json(G/(gate+'.json'),result)
 report=['# Gate '+gate,status,'Elapsed seconds: %.2f'%result['elapsed_seconds'],'', 'Error: '+str(error) if error else 'All checks passed.','',json.dumps(details,indent=2,ensure_ascii=False),'','| Check | Passed |','|---|---|']+[f"| {x['name']} | {x['passed']} |" for x in checks]
 (G/(gate+'.md')).write_text('\n'.join(report)+'\n');print(gate,status,flush=True)
 return 0 if status=='PASSED' else 1
if __name__=='__main__':sys.exit(main())
