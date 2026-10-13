"""Paired analysis of the completed seed-42 SV100 camera warm-up runs."""
import csv
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/poseopt_sv100'
NEW = ROOT/'results/poseopt_sv100_warmup2500'
OUT = NEW/'analysis'
OUT.mkdir(exist_ok=True)

def read_rows(path):
    return {(r['method'], r['case']): r for r in csv.DictReader((path/'summary.csv').open())}

a, b = read_rows(OLD), read_rows(NEW)
cases = [c for m, c in a if m == 'naive']
metrics = ['psnr', 'ssim', 'lpips', 'raw_rotation_deg', 'aligned_rotation_deg',
           'raw_translation_pct', 'aligned_translation_pct']
paired = []
for method in ('naive', 'camp'):
    for case in cases:
        key = method, case
        assert a[key]['status'] == b[key]['status'] == 'SUCCESS'
        row = dict(method=method, case=case)
        for metric in metrics:
            row[metric+'_before'] = float(a[key][metric])
            row[metric+'_after'] = float(b[key][metric])
            row[metric+'_delta'] = row[metric+'_after']-row[metric+'_before']
        row['psnr_delta_fixed_after'] = float(b[key]['psnr_delta_fixed'])
        paired.append(row)
with (OUT/'paired_metrics.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)

aggregate = {}
for method in ('naive', 'camp'):
    rows = [r for r in paired if r['method'] == method]
    aggregate[method] = {}
    for metric in metrics:
        aggregate[method][metric] = {
            'before': st.mean(r[metric+'_before'] for r in rows),
            'after': st.mean(r[metric+'_after'] for r in rows),
            'delta': st.mean(r[metric+'_delta'] for r in rows),
            'improved': sum(r[metric+'_delta'] > 0 if metric in ('psnr','ssim')
                            else r[metric+'_delta'] < 0 for r in rows)}
(OUT/'aggregate_metrics.json').write_text(json.dumps(aggregate, indent=2)+'\n')

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

fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for col,method in enumerate(('naive','camp')):
    for directory,label,style in ((OLD,'no warm-up','--'),(NEW,'warm-up','-')):
        state=json.loads((directory/'state.json').read_text())
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

lines=['# Warm-up 配对实验分析','',
'范围：Lego 单场景、SV100 九种扰动条件、seed 42、每组 30,000 步。18/18 个实验成功。',
'已核对两轮 manifest：数据、固定相机基线、环境一致，仅代码指纹不同。18 对保存配置仅 warmup_steps（0→2500）及输出路径不同。',
'','## 核心结果','',
'naive 的 9 个条件在 PSNR、SSIM、LPIPS 上全部改善；CamP 的 8 个条件在三个指标上改善，T5 三个指标均退化。',
'','| 方法 | PSNR 前→后 | ΔPSNR | SSIM 前→后 | LPIPS 前→后 | PSNR 改善条件 |',
'|---|---:|---:|---:|---:|---:|']
for m in ('naive','camp'):
    v=aggregate[m]
    lines.append(f"| {m} | {v['psnr']['before']:.3f} → {v['psnr']['after']:.3f} | {v['psnr']['delta']:+.3f} | {v['ssim']['before']:.4f} → {v['ssim']['after']:.4f} | {v['lpips']['before']:.4f} → {v['lpips']['after']:.4f} | {v['psnr']['improved']}/9 |")
lines += ['', '下列是条件间算术平均，不能作为跨场景或跨种子统计显著性证据。','',
'| 条件 | naive PSNR 前→后 | Δ | CamP PSNR 前→后 | Δ |','|---|---:|---:|---:|---:|']
for c in cases:
    n=next(r for r in paired if r['method']=='naive' and r['case']==c)
    k=next(r for r in paired if r['method']=='camp' and r['case']==c)
    lines.append(f"| {c} | {n['psnr_before']:.2f} → {n['psnr_after']:.2f} | {n['psnr_delta']:+.2f} | {k['psnr_before']:.2f} → {k['psnr_after']:.2f} | {k['psnr_delta']:+.2f} |")
lines += ['', '![重建质量](quality_comparison.png)', '', '## 退化是否消失？','',
'没有。两种方法优于固定相机基线的 PSNR 条件数均从 4/9 增至 7/9；GT 和 T0.5 仍低于固定相机。',
'GT：固定相机 27.12 dB，naive + warm-up 22.71 dB，CamP + warm-up 21.08 dB，分别相差 -4.41 和 -6.04 dB。',
'T0.5 的 PSNR/SSIM 低于固定相机，但 LPIPS 更好，因此不能笼统称为所有视觉指标均退化。',
'warm-up 后 naive 在 8/9 条件的 PSNR 优于 CamP；只有 R2 的 CamP 高 0.089 dB。这个排序只适用于本实现与本设置。',
'', '## 早期相机漂移','',
'GT 初始化提供了最直接的诊断：初始相机本来正确，早期偏离可直接观察。第 1,000 步：',
'| 方法 | 原始旋转误差（°）前→后 | 原始平移误差（场景半径 %）前→后 |',
'|---|---:|---:|','| naive | 1.626 → 0.647 | 1.142 → 0.368 |','| CamP | 4.870 → 1.206 | 9.017 → 2.649 |','',
'![早期漂移](gt_early_pose_drift.png)','',
'这些结果支持“减弱早期相机更新可抑制漂移”的解释，但未证明场景未学好是唯一原因。该实验仍从第一步更新相机，未测试完全冻结后的延迟启动。',
'', '## 位姿精度与评估坐标','',
'naive 对齐后平均旋转误差从 0.2036° 降至 0.1408°，平均平移误差从 0.2568% 降至 0.2114%。',
'CamP 对齐后平均旋转误差几乎不变（0.1313°→0.1310°），平移误差略升（0.2669%→0.2685%）。CamP 的渲染改善不能简单等同于位姿精度全面改善。',
'本实现只在训练分支修正相机，测试视角保持原始坐标。Sim(3) 对齐只用于位姿诊断，没有作用于渲染评估。',
'因此 PSNR 的变化可能混合场景质量变化与全局坐标漂移。CamP/R1 的原始平移误差从 1.015% 升至 3.250%，但对齐后从 0.268% 降至 0.257%，提示这一区分非常重要。',
'CamP/T5 的 PSNR 下降 0.986 dB，原始平移误差 0.741%→1.642%，而对齐后 0.280%→0.285%；全局漂移是待验证解释，不能据此直接确认为原因。',
'', '## 结论边界与下一步','',
'1. 将 warm-up 作为后续 naive 基线；当前证据表明它明显改善这一个场景的多个扰动条件。',
'2. 先用已有 checkpoint 补充统一坐标对齐的 held-out 评估，以分辨坐标漂移与实际场景质量；同时保留原始坐标评估。无需先重训。',
'3. 在 GT、T1、T5 等代表条件补做至少 3 个配对随机种子，检验跨运行稳定性；固定初始化和同一组数据扰动。',
'4. 再比较无 warm-up、warm-up、降低相机学习率，以及固定前 K 步的 delay；避免一次改多个因素。',
'5. 本次内置日程在 warm-up 后用剩余 27,500 步衰减，所以后期学习率也与原实验不同；它验证了完整日程调整的效果，尚未完全隔离早期阶段的因果作用。',
'', '推荐表述：在 Lego SV100、seed 42 的配对实验中，相机学习率 warm-up 改善了 naive 的全部条件和 CamP 的大多数条件，并降低了正确相机初始化下的早期漂移，但未消除低噪声条件的性能退化，也未证明跨种子稳定性。','']
(OUT/'report.md').write_text('\n'.join(lines))
print('Wrote analysis artifacts to',OUT)
