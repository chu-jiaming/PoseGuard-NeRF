"""Read-only progress monitor for the delivery controller."""
import json,time
from pathlib import Path
D=Path(__file__).resolve().parents[1]
while True:
 rows=[json.loads(x) for x in (D/'render_records.jsonl').read_text().splitlines()]
 videos=[r for r in rows if r['video']];r=rows[-1]
 print(json.dumps(dict(stills=len(rows)-len(videos),video_frames=len(videos),condition=r['condition'],method=r['method'],mode=r['mode'],last_view=r['view'],video=r['video'],seconds=round(r['render_seconds'],3))),flush=True)
 control=(D/'completion.log').read_text()
 if (D/'STOP.json').exists() or 'Traceback' in control:
  print('FAILED',control[-2000:],flush=True);break
 if control.rstrip().endswith('COMPLETE'):
  print('ALL COMPLETE',flush=True);break
 if 'export_curves.py' in control and (D/'curves.log').exists():print('CURVES',(D/'curves.log').read_text().splitlines()[-2:],flush=True)
 time.sleep(45)
