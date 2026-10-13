#!/usr/bin/env python
"""Publish a compact interpretation and plot for a finished diagnostic run."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    with (args.directory/'summary.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=5:raise ValueError('Wait for all five variants')
    by={r['method']:r for r in rows}
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    names=[r['method'].replace('_warmup',' + WU') for r in rows];x=np.arange(5)
    for ax,metric in zip(axes,('psnr','ssim','lpips')):
        for offset,mode in [(-.19,'raw'),(.19,'aligned')]:
            ax.bar(x+offset,[float(r[mode+'_'+metric]) for r in rows],width=.38,label=mode)
        ax.set_xticks(x,names,rotation=35,ha='right');ax.set_title(metric.upper()+(' (lower is better)' if metric=='lpips' else ' (higher is better)'))
        ax.grid(axis='y',alpha=.2);ax.legend()
    fig.savefig(args.directory/'metrics_comparison.png',dpi=180);plt.close(fig)
    fixed=float(by['fixed']['aligned_psnr'])
    lines=['# GT 条件评估诊断','',
           '范围、分辨率及评估视角见 [协议与结果表](report.md)。以下数字仅在该次评估协议内比较。',
           '只拟合训练相机中心的全局 Sim(3)，不使用测试图像拟合、不更新网络。','',
           '| 方法 | 原坐标 PSNR | 对齐后 PSNR | 对齐提升 | 对齐后相对 fixed |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        raw,aligned=float(r['raw_psnr']),float(r['aligned_psnr'])
        lines.append('| %s | %.3f | %.3f | %+.3f | %+.3f |'%(r['method'],raw,aligned,aligned-raw,aligned-fixed))
    lines += ['', '## Warm-up 的配对收益','']
    for method in ('naive','camp'):
        base,wu=by[method],by[method+'_warmup']
        raw=float(wu['raw_psnr'])-float(base['raw_psnr']);aligned=float(wu['aligned_psnr'])-float(base['aligned_psnr'])
        lines.append('- %s：原坐标收益 %+.3f dB；对齐后收益 %+.3f dB。'%(method,raw,aligned))
    lines += ['', '## 解释边界','',
              '- 对齐收益说明原坐标评估对全局漂移敏感，不能将全部指标下降直接归因为场景本身变差。',
              '- 对齐后与 fixed 的差距是残留性能差距；单凭此实验无法在局部位姿误差、场景拟合不足等原因之间做因果判定。',
              '- 中心拟合不保证相机朝向或图像误差最优；这是诊断性对齐，不是利用测试图像求出的最佳结果。',
              '- dB 差值不能用来严格分解各原因的误差占比，也不能作为跨场景、跨种子稳定性的证据。','',
              '![指标对比](metrics_comparison.png)','']
    if (args.directory/'comparison_000.png').exists():lines.append('![相同视角对齐前后](comparison_000.png)')
    (args.directory/'findings.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
