"""Serial inference jobs; stop immediately on any failed self-check or worker error."""
import os,sys,json,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parents[1];env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
(D/'logs').mkdir(exist_ok=True)
def run(condition,method,mode='raw',video=False):
 label='_'.join([condition,method,mode,'video' if video else 'still']);cmd=[sys.executable,str(D/'scripts/render.py'),'--condition',condition,'--method',method,'--mode',mode]+(['--video'] if video else [])
 print('START',label,flush=True)
 with (D/'logs'/(label+'.log')).open('a') as f:
  code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 if code:raise RuntimeError('STOP: failed '+label+'; see log')
 print('DONE',label,flush=True)
if __name__=='__main__':
 stage=sys.argv[1]
 if (D/'STOP.json').exists():raise RuntimeError('STOP.json exists; review required')
 methods=['fixed','naive','camp','naive_warmup','camp_warmup']
 if stage=='P0':
  for m in methods:
   run('SV100-GT',m)
   if m!='fixed':run('SV100-GT',m,'aligned')
 elif stage=='P1':
  for c in ['SV100-T5','SV100-R5','SV100-T1']:
   for m in methods:run(c,m)
 elif stage=='P2':
  if json.loads((D/'timing_estimate.json').read_text())['exceeds_8_hours']:raise RuntimeError('P2 skipped by 8-hour rule')
  for c,m,mode in [('SV100-GT','fixed','raw'),('SV100-GT','naive_warmup','aligned'),('SV100-T5','fixed','raw'),('SV100-T5','naive_warmup','raw')]:run(c,m,mode,True)
 else:raise ValueError(stage)
 print(stage+' COMPLETE',flush=True)
