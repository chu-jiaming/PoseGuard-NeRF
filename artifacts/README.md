# 实验产物索引

本目录保存已完成实验的小型静态档案。单次实验按“实验类别 → 方法 → 位姿条件”组织，汇总分析与运行记录单独存放。

| 目录 | 内容 |
|---|---|
| `experiments/baseline/fixed/SV100-*/` | 9 次固定相机训练的配置、指标、数据解析变换和元数据 |
| `experiments/pose_optimization/{naive,camp}/SV100-*/` | 18 次无 warm-up 训练的配置、指标、优化位姿、预条件诊断和完成标记 |
| `experiments/warmup/{naive,camp}/SV100-*/` | 18 次 warm-up 训练的同类资料 |
| `experiments/alignment_full/<方法>/SV100-GT/` | GT 条件五种设置的完整对齐评估变换 |
| `experiments/alignment_preview/<方法>/SV100-GT/` | GT 条件五种设置的预览对齐评估变换 |
| `analysis/<实验类别>/` | 汇总表、分析报告、对比图；`pose_audit/` 保存 36 组位姿审计的汇总与变换集合 |
| `records/<实验类别>/` | 批次级状态、manifest 和环境记录 |

对齐评估的方法目录为 `fixed`、`naive`、`camp`、`naive_warmup`、`camp_warmup`。`SV100-GT` 表示正确位姿初始化；`SV100-T*` 和 `SV100-R*` 分别表示平移与旋转扰动条件。

## 汇总与对比分析

| 内容 | 入口 |
|---|---|
| 固定相机基线 | [9 条件汇总](analysis/baseline/SV100_full_summary.csv)、[平移对照](analysis/baseline/SV100_translation_sweep.csv)、[旋转对照](analysis/baseline/SV100_rotation_sweep.csv) |
| 首轮三方法比较 | [27 行汇总](analysis/pose_optimization/summary.csv)、[历史报告](analysis/pose_optimization/report.md.txt)、[Naive 失败分析](analysis/pose_optimization/naive_failure_analysis.md.txt) |
| Warm-up 配对比较 | [汇总](analysis/warmup/summary.csv)、[配对差值](analysis/warmup/paired_metrics.csv)、[统计](analysis/warmup/aggregate_metrics.json)、[阶段报告](analysis/warmup/report.md) |
| GT 完整对齐评估 | [汇总](analysis/alignment_full/summary.csv)、[报告](analysis/alignment_full/report.md)、[结论](analysis/alignment_full/findings.md) |
| GT 预览对齐评估 | [汇总](analysis/alignment_preview/summary.csv)、[报告](analysis/alignment_preview/report.md)、[结论](analysis/alignment_preview/findings.md) |
| 全条件位姿审计 | [36 组汇总](analysis/pose_audit/summary.csv)、[变换集合](analysis/pose_audit/transforms.json) |

## 配置与图片入口

| 内容 | 入口 |
|---|---|
| Naive GT 配置对照 | [无 warm-up](experiments/pose_optimization/naive/SV100-GT/config.yml)、[有 warm-up](experiments/warmup/naive/SV100-GT/config.yml) |
| 优化位姿示例 | [Naive GT + warm-up](experiments/warmup/naive/SV100-GT/optimized_poses.json) |
| Warm-up 对比图 | [重建指标](analysis/warmup/quality_comparison.png)、[早期漂移](analysis/warmup/gt_early_pose_drift.png) |
| 对齐指标图 | [完整评估](analysis/alignment_full/metrics_comparison.png)、[低分辨率预览](analysis/alignment_preview/metrics_comparison.png) |
| 对齐前后预览 | [视角 0](analysis/alignment_preview/comparison_000.png)、[视角 50](analysis/alignment_preview/comparison_050.png)、[视角 100](analysis/alignment_preview/comparison_100.png) |

## 读取与重新分析

在仓库根目录运行 `uv run python scripts/analysis/analyze_warmup_results.py`，可从归档 CSV 重新计算配对统计，默认输出到 `results/published_analysis/`。加 `--plots` 可重画质量对比图，需要 NumPy 和 Matplotlib。

发布档案不含原始数据、checkpoint、TensorBoard、原始渲染数组或完整训练日志。重新渲染需要原模型和数据；重画早期漂移需要原服务器的 `diagnostics.jsonl`。完整评估使用 200 张 800×800 图，预览使用 20 张 200×200 图，两者的指标应分开比较。

历史报告 `.md.txt` 保留当时文字及未打包图片的路径。Warm-up 阶段报告早于对齐诊断，其中 GT 对齐的后续工作已完成，多种子实验未执行；后续对齐解释见对应 findings。

`records/`、配置及完成标记中的服务器绝对路径属于历史来源信息，不能在新机器直接用于恢复。所有现存记录均保留，目录整理没有重新训练或重新计算图像指标。通过 Git status 和 Git diff 查看档案变更。
