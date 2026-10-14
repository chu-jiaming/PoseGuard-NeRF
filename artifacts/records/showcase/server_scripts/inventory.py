import os,json,re,hashlib,time
from pathlib import Path
import yaml
ROOT=Path('/root/workspace/poseguard-nerf'); OUT=ROOT/'delivery_20261014'
def parse(node):
 if isinstance(node,yaml.MappingNode): return {k.value:parse(v) for k,v in node.value}
 if isinstance(node,yaml.SequenceNode):
  vals=[parse(x) for x in node.value]
  return str(Path(*vals)) if 'pathlib' in node.tag else vals
 return node.value
expected={'SV100-GT':[27.117,17.909,19.900,22.707,21.076],'SV100-T5':[14.145,17.534,19.717,22.681,18.731],'SV100-R5':[12.007,17.470,15.084,18.334,16.247]}
methods=['fixed','naive','camp','naive_warmup','camp_warmup']; candidates=[]
metrics={}
for base in [ROOT/'results']:
 for directory,dirs,files in os.walk(str(base),followlinks=True):
  if 'metrics.json' in files:
   p=Path(directory)/'metrics.json'
   try:
    m=json.loads(p.read_text()); ck=Path(m['checkpoint']); ck=ck if ck.is_absolute() else ROOT/ck
    metrics.setdefault(str(ck.resolve()),[]).append((p,m))
   except (KeyError,ValueError): pass
for base in [ROOT/'outputs',ROOT/'results']:
 for directory,dirs,files in os.walk(str(base),followlinks=True):
  if 'step-000029999.ckpt' not in files: continue
  ck=Path(directory)/'step-000029999.ckpt'; cfg=ck.parent.parent/'config.yml'
  if not cfg.exists(): continue
  c=parse(yaml.compose(cfg.read_text())); name=c['method_name']
  if name not in ['vanilla-nerf','poseguard-naive','poseguard-camp']: continue
  dm=c['pipeline']['datamanager']; data=c.get('data') if c.get('data') not in [None,'null'] else (dm.get('data') if dm.get('data') not in [None,'null'] else dm['dataparser']['data']); d=Path(data).name
  if d in ['lego','lego_sv100']: cond='SV100-GT'
  else:
   match=re.search(r'([tr])noise_(\d+(?:p\d+)?)(?:pct|deg)',d)
   if not match: continue
   cond='SV100-'+match[1].upper()+match[2].replace('p','.')
  warm=0 if name=='vanilla-nerf' else int(c['optimizers']['camera_opt']['scheduler']['warmup_steps'])
  method='fixed' if name=='vanilla-nerf' else name.replace('poseguard-','')+('_warmup' if warm==2500 else '')
  mm=metrics.get(str(ck.resolve()),[])
  psnrs=[float(m['results']['fine_psnr']) for p,m in mm]
  exp=expected.get(cond,[None]*5)[methods.index(method)]
  passed=bool(mm) and len(set(psnrs))==1 and (exp is None or abs(psnrs[0]-exp)<=.01) and warm in [0,2500]
  candidates.append(dict(condition=cond,method=method,config=str(cfg),checkpoint=str(ck),metrics=[str(p) for p,m in mm],PSNR=psnrs[0] if mm else None,expected_PSNR=exp,verification_passed=passed,data=data,warmup_steps=warm,seed=c['machine']['seed'],steps=c['max_num_iterations']))
rows=[]
for cond in ['SV100-GT']+['SV100-'+x+y for x in 'TR' for y in ['0.5','1','2','5']]:
 for method in methods:
  hits=[r for r in candidates if r['condition']==cond and r['method']==method]
  if len(hits)==1: rows.append(hits[0])
  else: rows.append(dict(condition=cond,method=method,verification_passed=False,status='missing' if not hits else 'ambiguous',candidates=hits))
for r in rows:
 if 'checkpoint' in r and (r['condition'],r['method']) in [('SV100-GT','fixed'),('SV100-T5','fixed')]:
  h=hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest(); r['sha256']=h
  target={'SV100-GT':'c3d47101d4df1e2039ba3aa1667a7c1c98c7d12fb2e1b5d89cb1b8d31c785a6e','SV100-T5':'89134af0a04380cea1744aa189003cc7a2b070a0ab636b2db8508254992b1795'}[r['condition']]
  r['sha256_passed']=h==target; r['verification_passed'] &= h==target
ready=all(r['verification_passed'] for r in rows if r['condition'] in expected)
(OUT/'inventory.json').write_text(json.dumps(dict(render_gate_passed=ready,runs=rows),indent=2)+'\n')
for r in rows: print(r['condition'],r['method'],r.get('PSNR'),r['verification_passed'],r.get('status',''))
print('GATE',ready)
