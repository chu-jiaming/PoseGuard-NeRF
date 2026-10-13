# 发布结果档案

本目录是已完成实验的小型静态快照，可以随 Git 克隆。原始运行仍位于本机 `results/`、`outputs/` 及其数据盘链接中；本目录不依赖这些链接。

| 目录 | 内容 |
|---|---|
| `historical/` | 首轮 Fixed/Naive/CamP 的已跟踪小型记录、配置、指标、位姿、状态和环境 |
| `warmup/` | 第二轮 18 次训练的配置、指标、位姿、状态、配对分析与图 |
| `alignment_gt_full/` | GT 条件五种设置的完整测试集对齐结果 |
| `alignment_gt_preview/` | 独立低分辨率预览结果及三个代表视角拼图 |
| `alignment_pose_audit/` | 36 组优化位姿的审计及源文件指纹 |

[manifest.json](manifest.json) 记录原始相对路径、档案路径、大小和 SHA-256。复制时保持原字节不变。历史 Markdown 报告以 `.md.txt` 保存，保留当时文字和未打包的本地图片路径；它们属于历史记录，最新解释以根目录 README 和完整对齐 findings 为准。

Warm-up 的 [原阶段分析](warmup/report.md) 写于渲染对齐之前，其中“下一步补充对齐”已经在 GT 条件完成；其多种子建议未执行。不要把历史建议误认为本项目当前必做事项。

配置和状态中的绝对路径、代码哈希、运行时间均是历史溯源信息。当前代码新增了路径配置，不与历史代码指纹相同；不要用历史状态在新机器直接 resume。

没有打包数据、checkpoint、TensorBoard、原始数组或全部渲染。要重新计算图像指标，需自行训练或另行取得原模型。历史 source_fingerprints 涉及未打包大文件，本仓库只能验证已归档文件。

```bash
python scripts/verify_artifacts.py
```
