#!/usr/bin/env python
"""CPU audit of all saved pose estimates; does not evaluate novel views."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poseguard.alignment import fit_sim3, transform_poses
from scripts.sv100_pipeline import GT


def main():
    gt=json.loads((GT/'transforms_train.json').read_text())
    reference=np.asarray([f['transform_matrix'][:3] for f in gt['frames']])
    radius=np.linalg.norm(reference[:,:,3],axis=1).mean()
    output=ROOT/'results/alignment_pose_audit';output.mkdir(exist_ok=True)
    rows=[]; transforms={}
    for directory,label in [('poseopt_sv100','original'),('poseopt_sv100_warmup2500','warmup')]:
        state=json.loads((ROOT/'results'/directory/'state.json').read_text())
        for key,entry in state['experiments'].items():
            if entry['status']!='SUCCESS':raise ValueError(key)
            config=Path(entry['config']).read_text()
            if '      scale_factor: 1.0\n' not in config:raise ValueError('Unexpected parser scale')
            poses=json.loads((Path(entry['attempt'])/'optimized_poses.json').read_text())
            estimated=np.asarray(poses['poses'])
            sim=fit_sim3(reference[:,:,3],estimated[:,:,3]); fitted=transform_poses(reference,sim)
            residual=np.linalg.norm(fitted[:,:,3]-estimated[:,:,3],axis=1).mean()
            raw=np.linalg.norm(reference[:,:,3]-estimated[:,:,3],axis=1).mean()
            r=np.asarray(sim['rotation'])
            angle=np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))
            method,case=key.split('/')
            rows.append(dict(variant=label,method=method,case=case,scale=sim['scale'],
                             global_rotation_deg=angle,global_translation_radius_pct=100*np.linalg.norm(sim['translation'])/radius,
                             raw_center_error_pct=100*raw/radius,postfit_center_residual_pct=100*residual/radius,
                             psnr=entry['metrics']['psnr']))
            transforms[label+'/'+key]=sim
    with (output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'transforms.json').write_text(json.dumps(transforms,indent=2)+'\n')
    print('Audited',len(rows),'saved pose sets')
    for r in rows:
        if r['case']=='SV100-GT':print(r)

if __name__=='__main__':main()
