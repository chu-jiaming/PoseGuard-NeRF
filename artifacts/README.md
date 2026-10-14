# 实验产物索引

本目录保存已完成实验的小型静态档案。单次实验按“实验类别 → 方法 → 位姿条件”组织，汇总分析与运行记录单独存放。

| 目录 | 内容 |
|---|---|
| `experiments/baseline/fixed/SV100-*/` | 9 次固定相机训练的配置、指标、数据解析变换和元数据 |
| `experiments/pose_optimization/{naive,camp}/SV100-*/` | 18 次无 warm-up 训练的配置、指标、优化位姿、预条件诊断和完成标记 |
| `experiments/warmup/{naive,camp}/SV100-*/` | 18 次 warm-up 训练的同类资料 |
| `experiments/alignment_full/<方法>/SV100-GT/` | GT 条件五种设置的对齐评估变换 |
| `analysis/<实验类别>/` | 汇总表、分析报告、对比图；`pose_audit/` 保存 36 组位姿审计的汇总与变换集合 |
| `records/<实验类别>/` | 批次级状态、manifest 和环境记录 |
| `analysis/training_curves/` | 45 次训练的精简 TensorBoard 标量（训练 loss、eval PSNR、相机学习率）与 36 份位姿诊断 `diagnostics/*.jsonl` |
| `analysis/showcase_renders/` | README 展示图的逐图指标（96 张 800×800 渲染）及误差图色标 |
| `records/showcase/` | 展示素材的服务器交付报告、run 清单、checkpoint 哈希和渲染脚本 |

对齐评估的方法目录为 `fixed`、`naive`、`camp`、`naive_warmup`、`camp_warmup`。`SV100-GT` 表示正确位姿初始化；`SV100-T*` 和 `SV100-R*` 分别表示平移与旋转扰动条件。

## 汇总与对比分析

| 内容 | 入口 |
|---|---|
| 固定相机基线 | [9 条件汇总](analysis/baseline/SV100_full_summary.csv) |
| 首轮三方法比较 | [27 行汇总](analysis/pose_optimization/summary.csv) |
| Warm-up 配对比较 | [汇总](analysis/warmup/summary.csv)、[配对差值](analysis/warmup/paired_metrics.csv)、[统计](analysis/warmup/aggregate_metrics.json)、[阶段报告](analysis/warmup/report.md) |
| GT 完整对齐评估 | [汇总](analysis/alignment_full/summary.csv)、[报告](analysis/alignment_full/report.md)、[结论](analysis/alignment_full/findings.md) |
| 全条件位姿审计 | [36 组汇总](analysis/pose_audit/summary.csv)、[变换集合](analysis/pose_audit/transforms.json) |

## 配置与图片入口

| 内容 | 入口 |
|---|---|
| Naive GT 配置对照 | [无 warm-up](experiments/pose_optimization/naive/SV100-GT/config.yml)、[有 warm-up](experiments/warmup/naive/SV100-GT/config.yml) |
| 优化位姿示例 | [Naive GT + warm-up](experiments/warmup/naive/SV100-GT/optimized_poses.json) |
| Warm-up 对比图 | [重建指标](analysis/warmup/quality_comparison.png)、[早期漂移](analysis/warmup/gt_early_pose_drift.png) |
| 对齐指标图 | [完整评估](analysis/alignment_full/metrics_comparison.png) |

## 展示素材

README 中的渲染图和视频由服务器在同一批 checkpoint 上推理生成，没有新训练。推理时 near 按 Nerfstudio 默认行为重置为 0，far 为 6，aligned 渲染的 far 乘以拟合尺度，与历史评估一致。aligned 静态图与历史逐图指标逐张核对，偏差 0 dB；raw 静态图只做了 4 视角均值与全量均值的比对，最大差 3.6 dB。服务器 Gate B 的 ffprobe 检查项误报为通过，视频参数已在本地用 ffprobe 复核（H.264、yuv420p、30 fps、200 帧）。原始图片与视频不入库，导入与作图方法见 [运行指南](../docs/RUNNING.md#7-showcase-figures)。

## 读取与重新分析

在仓库根目录运行 `uv run python scripts/analysis/analyze_warmup_results.py`，可从归档 CSV 重新计算配对统计，默认输出到 `results/published_analysis/`。加 `--plots` 可重画质量对比图，加 `--diagnostics` 可用 `analysis/training_curves/diagnostics/` 中归档的 GT 轨迹重画早期漂移图，两者都需要 NumPy 和 Matplotlib。

发布档案不含原始数据、checkpoint、TensorBoard event 文件、原始渲染数组或完整训练日志；TensorBoard 标量只保留了精简后的 `analysis/training_curves/tb_scalars.csv`。重新渲染需要原模型和数据，7 个最终 checkpoint 见 [v1.0.0 Release](https://github.com/chu-jiaming/PoseGuard-NeRF/releases/tag/v1.0.0)。对齐评估使用全部 200 张 800×800 测试图。

Warm-up 阶段报告早于对齐诊断，其中 GT 对齐的后续工作已完成，多种子实验未执行；后续对齐解释见对应 findings。

`records/showcase/DELIVERY_REPORT.md` 是服务器交付报告原文：其中“CSV标签行数”统计的是服务器上的完整标量 CSV，不是仓库里的精简版；提到的 `gates/`、`archive_receipt.json` 只在服务器交付包中。

`records/`、配置及完成标记中的服务器绝对路径属于历史来源信息，不能在新机器直接用于恢复。档案中的指标均来自原始评估，整理时没有重新训练或重新计算。
