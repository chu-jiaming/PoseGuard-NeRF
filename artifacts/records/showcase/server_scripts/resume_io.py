import json,os,fcntl
from pathlib import Path
from PIL import Image
D=Path(__file__).resolve().parents[1]
def atomic_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def png(im,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');im.save(t,format='PNG');os.replace(t,p)
def valid(p,size):
 try:
  with Image.open(p) as im:im.load();return im.size==size and im.mode=='RGB'
 except (OSError,ValueError):return False
def records():
 p=D/'render_records.jsonl';out={}
 if p.exists():
  for line in p.read_text().splitlines():
   try:r=json.loads(line);out[r['file']]=r
   except (ValueError,KeyError):continue
 return list(out.values())
def save_record(r):
 # Record first: an interrupted PNG write can then be safely retried.
 atomic_json(D/'frame_records'/Path(r['file']).parent.name/(Path(r['file']).stem+'.json'),r)
 rows={v['file']:v for v in records()};rows[r['file']]=r
 p=D/'render_records.jsonl';t=p.with_name(p.name+'.tmp');t.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in rows.values()));os.replace(t,p)
def lock():
 inherited=os.environ.get('POSEGUARD_LOCK_FD')
 if inherited:
  fd=int(inherited);assert os.fstat(fd).st_ino==(D/'.render.lock').stat().st_ino;return fd
 fd=os.open(str(D/'.render.lock'),os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return fd
