# 复现与环境

## 1. 两种使用方式

- **查看已完成的实验**：直接阅读 README 和 `artifacts/`，不需要安装模型依赖。可运行 `python scripts/verify_artifacts.py` 校验发布档案。
- **重新训练**：从干净 checkout 开始准备数据、运行 Fixed，再运行两个位姿优化日程，最后执行诊断。45 次训练需要较长 GPU 时间，本次发布整理没有重新训练。

发布包不含 checkpoint。历史配置、状态和 CSV 保留原机器路径用于追溯；不要复制到运行目录冒充可恢复状态。重训会自动生成当前机器的路径、指纹和配置。迁移旧权重的路径修复不由当前入口自动完成。

## 2. 环境

历史环境：Linux、Python 3.8.10、PyTorch 2.1.2+cu118、torchvision 0.16.2+cu118、Nerfstudio 1.1.5。主要版本约束见 [requirements.txt](requirements.txt)，原始完整环境快照见 [environment.txt](artifacts/historical/results/poseopt_sv100/environment.txt)。原始快照包含 Conda 本地构建路径，是审计记录，不是直接可安装的 lockfile。

创建并激活独立环境，在配置 CUDA 驱动与所需构建工具后安装：

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
PYTHONPATH=. python -m unittest discover -s tests -v
```

以上为安装入口，不保证任意新平台上的依赖解析成功；本次只验证已有实验环境，未做全新环境安装。Nerfstudio 的可选 CUDA 扩展与构建要求取决于安装环境。本项目模型为 vanilla-NeRF。

脚本使用激活环境的 `python`、`ns-train`、`ns-eval`，三者应来自同一环境。Shell 编排还需 `bash`、`flock`、`pgrep`；后台运行可用 `tmux`。

## 3. 数据

自行准备 NeRF-Synthetic 数据；Lego 应有 `train/val/test` 图片和对应 `transforms_*.json`。数据不随仓库分发，按原数据来源的使用条款获取。

```bash
export POSEGUARD_DATA=/absolute/path/to/nerf_synthetic
```

该目录下应有 `lego/`。不设置时默认为仓库同级的 `data/nerf_synthetic`。先生成八种扰动（仅对尚不存在的目标目录操作）：

```bash
python scripts/prepare_datasets.py
```

生成器按 seed 42 调用 `perturb_poses.py`，复用图片软链接，保持验证/测试相机不变。目标目录已存在时会拒绝覆盖。

## 4. 从头运行

在仓库根目录、同一激活环境依次执行。固定基线入口拒绝覆盖已有运行目录；在已有本地实验的工作区，使用新的 checkout，不要删除历史数据。

```bash
bash scripts/run_full_sv100_pipeline.sh
bash scripts/run_pose_optimization_pipeline.sh --preflight
bash scripts/run_pose_optimization_pipeline.sh --dry-run
bash scripts/run_pose_optimization_pipeline.sh
bash scripts/run_warmup_experiments.sh
```

Fixed 产生 9 次训练；无 warm-up 和 warm-up 各产生 18 次训练。Warm-up 复用 Fixed，不再训练它。查看、停止和恢复见 [运行说明](POSE_OPTIMIZATION.md) 与 [warm-up 说明](WARMUP_EXPERIMENTS.md)。可将各条训练命令放入 tmux 中运行。

`--preflight`、`--dry-run` 和现有 `--smoke` 都要求完整的 Fixed 基线及其 checkpoint；它们不是空仓库的免数据验证入口。免数据检查请运行单元测试。

## 5. 分析与对齐

上述运行全部完成后：

```bash
python scripts/analyze_warmup_results.py
python scripts/audit_pose_alignment.py
bash scripts/run_alignment_diagnostic.sh --output results/alignment_gt_preview
python scripts/summarize_alignment.py results/alignment_gt_preview
bash scripts/run_alignment_diagnostic.sh --output results/alignment_gt_full --full
python scripts/summarize_alignment.py results/alignment_gt_full
```

对齐渲染入口当前仅支持 GT 条件。完整评估为 200 张 800×800 图；预览为固定 20 张 200×200 图。两者不能直接比较数值。新运行产物写入被忽略的 `results/`、`outputs/`、`logs/`，不会覆盖 `artifacts/` 中的发布快照。

## 6. 恢复限制

运行时修改代码、环境、数据或超参数会改变指纹。发布整理修改了路径配置，所以历史运行不能直接在当前代码下继续恢复；已有完成结果仍是原实验结果，发布档案保持原字节与原指纹。需要新实验时使用新目录和新 manifest。恢复不会保证随机数序列与不中断训练逐位一致。
