"""Recompute paired metrics from published CSVs; plotting and runtime traces are optional."""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['psnr', 'ssim', 'lpips', 'raw_rotation_deg', 'aligned_rotation_deg',
           'raw_translation_pct', 'aligned_translation_pct']


def read_rows(path):
    with (path/'summary.csv').open() as stream:
        return {(r['method'], r['case']): r for r in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=ROOT/'artifacts/analysis/pose_optimization')
    parser.add_argument('--warmup', type=Path, default=ROOT/'artifacts/analysis/warmup')
    parser.add_argument('--output', type=Path, default=ROOT/'results/published_analysis')
    parser.add_argument('--plots', action='store_true', help='Draw quality comparison; requires NumPy and Matplotlib')
    parser.add_argument('--diagnostics', action='store_true', help='Also plot runtime traces referenced by state.json; requires original server files')
    args = parser.parse_args()
    original, warmup, output = args.original.resolve(), args.warmup.resolve(), args.output.resolve()
    if output == ROOT/'artifacts' or ROOT/'artifacts' in output.parents:
        parser.error('Choose an output outside artifacts/ to preserve published results')
    a, b = read_rows(original), read_rows(warmup)
    cases = [c for m, c in a if m == 'naive']
    paired = []
    for method in ('naive', 'camp'):
        for case in cases:
            key = method, case
            if a[key]['status'] != 'SUCCESS' or b[key]['status'] != 'SUCCESS':
                raise ValueError('Incomplete pair: '+str(key))
            row = dict(method=method, case=case)
            for metric in METRICS:
                row[metric+'_before'] = float(a[key][metric])
                row[metric+'_after'] = float(b[key][metric])
                row[metric+'_delta'] = row[metric+'_after']-row[metric+'_before']
            row['psnr_delta_fixed_after'] = float(b[key]['psnr_delta_fixed'])
            paired.append(row)
    aggregate = {}
    for method in ('naive', 'camp'):
        rows = [r for r in paired if r['method'] == method]
        aggregate[method] = {}
        for metric in METRICS:
            aggregate[method][metric] = {
                'before': st.mean(r[metric+'_before'] for r in rows),
                'after': st.mean(r[metric+'_after'] for r in rows),
                'delta': st.mean(r[metric+'_delta'] for r in rows),
                'improved': sum(r[metric+'_delta'] > 0 if metric in ('psnr','ssim')
                                else r[metric+'_delta'] < 0 for r in rows)}
    output.mkdir(parents=True, exist_ok=True)
    with (output/'paired_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    (output/'aggregate_metrics.json').write_text(json.dumps(aggregate, indent=2)+'\n')
    if args.plots or args.diagnostics:
        plot_quality(paired, cases, b, output)
    if args.diagnostics:
        plot_drift(original, warmup, output)
    lines = ['# Warm-up 配对分析', '',
             '从输入 summary.csv 重新计算差值与条件平均；不加载模型，不重新渲染图像。',
             '原始输入：`'+str(original)+'`；warm-up 输入：`'+str(warmup)+'`。', '',
             '| 方法 | PSNR 前→后 | 平均 ΔPSNR | PSNR 改善条件 | SSIM 改善条件 | LPIPS 改善条件 |',
             '|---|---:|---:|---:|---:|---:|']
    for method, values in aggregate.items():
        psnr = values['psnr']
        lines.append(f"| {method} | {psnr['before']:.3f} → {psnr['after']:.3f} | {psnr['delta']:+.3f} | {psnr['improved']}/{len(cases)} | {values['ssim']['improved']}/{len(cases)} | {values['lpips']['improved']}/{len(cases)} |")
        print(f"{method}: mean PSNR delta {psnr['delta']:+.3f} dB; improved {psnr['improved']}/{len(cases)}")
    lines += ['', '| 方法 | 条件 | PSNR 前 | PSNR 后 | ΔPSNR |', '|---|---|---:|---:|---:|']
    for row in paired:
        lines.append('| {method} | {case} | {psnr_before:.3f} | {psnr_after:.3f} | {psnr_delta:+.3f} |'.format(**row))
    if args.plots or args.diagnostics:
        lines += ['', '![质量比较](quality_comparison.png)']
    if args.diagnostics:
        lines += ['', '![早期相机漂移](gt_early_pose_drift.png)']
    lines += ['', '## 如何解释', '',
              '- 发布输入是 Lego、SV100、seed 42 的原坐标测试结果；条件平均不是跨种子显著性证据。',
              '- PSNR、SSIM 越高越好；LPIPS 越低越好。此处按指标分别统计改善条件数。',
              '- warm-up 改变整个相机学习率日程，不能只归因为早期阶段。',
              '- GT 的完整渲染对齐诊断见仓库 artifacts/analysis/alignment_full/；其结果与此处原坐标指标分开解读。',
              '- 本脚本没有验证历史输入是否完全受控，也没有重新计算图像级指标。', '']
    (output/'report.md').write_text('\n'.join(lines))
    print('Wrote analysis to', output)


def plot_quality(paired, cases, b, OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
    x=np.arange(len(cases));colors={'naive':'#1976b9','camp':'#d87a21'}
    for method, offset in (('naive', -.19), ('camp', .19)):
        rows=[r for r in paired if r['method']==method]
        axes[0].bar(x+offset,[r['psnr_delta'] for r in rows],.36,label=method,color=colors[method])
        for suffix,style,label in (('before','--','no warm-up'),('after','-', 'warm-up')):
            axes[1].plot(x,[r['psnr_'+suffix] for r in rows],style,marker='o',markersize=4,
                         color=colors[method],label=f'{method}: {label}')
    axes[1].plot(x,[float(b['fixed',c]['psnr']) for c in cases],':',color='#444444',label='fixed cameras')
    axes[0].axhline(0,color='gray',lw=.8)
    axes[0].set(title='Warm-up benefit by condition',ylabel='PSNR change (dB)')
    axes[1].set(title='Final held-out reconstruction quality',ylabel='PSNR (dB)')
    for ax in axes:
        ax.set_xticks(x,[c.replace('SV100-', '', 1) for c in cases],rotation=35)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.suptitle('Lego SV100 | seed 42 | 30,000 steps | camera warm-up: 0 vs 2,500 steps')
    fig.savefig(OUT/'quality_comparison.png',dpi=180);plt.close(fig)



def plot_drift(OLD, NEW, OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,method in enumerate(('naive','camp')):
        for directory,label,style in ((OLD,'no warm-up','--'),(NEW,'warm-up','-')):
            state_directory = directory
            if directory.parent == ROOT/'artifacts/analysis':
                state_directory = ROOT/'artifacts/records'/directory.name
            state=json.loads((state_directory/'state.json').read_text())
            trace=Path(state['experiments'][method+'/SV100-GT']['attempt'])/'diagnostics.jsonl'
            rows={d['step']:d for d in map(json.loads,trace.read_text().splitlines())}
            data=[rows[k] for k in sorted(rows) if k<=10000]
            for row,(metric,unit) in enumerate((('rotation_deg_mean','Rotation error (degrees)'),
                                               ('translation_radius_pct_mean','Translation error (% scene radius)'))):
                axes[row,col].plot([d['step'] for d in data],
                    [d['pose_errors']['raw'][metric] for d in data],style,label=label)
                axes[row,col].set(xlabel='Training step',ylabel=unit,title=f'{method}: GT initialization')
                axes[row,col].grid(alpha=.2);axes[row,col].legend()
    fig.suptitle('Early camera drift from correct initial poses (raw errors; sampled every 1,000 steps)')
    fig.savefig(OUT/'gt_early_pose_drift.png',dpi=180);plt.close(fig)



if __name__ == '__main__':
    main()
