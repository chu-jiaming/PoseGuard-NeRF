import json
from pathlib import Path
D=Path(__file__).resolve().parents[1]
p=D/'render_records.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
print('stills',sum(not r['video'] for r in rows),'video_frames',sum(r['video'] for r in rows))
if rows:
 r=rows[-1];print({k:r[k] for k in ['condition','method','mode','view','video','metrics']})
for name in ['P0.log','P1.log','P2.log','completion.log']:
 p=D/name
 if p.exists():print(name,p.read_text().splitlines()[-2:])
if (D/'STOP.json').exists():print('STOP', (D/'STOP.json').read_text())
