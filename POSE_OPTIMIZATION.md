# PoseGuard：独立串行位姿优化实验

入口：`scripts/run_pose_optimization_pipeline.sh`。运行依赖本机已有的 `/root/miniconda3` Python 环境、Nerfstudio 1.1.5 和 CUDA；不需要 Codex 持续在线。

## 正式运行

```bash
cd /root/workspace/poseguard-nerf
bash scripts/run_pose_optimization_pipeline.sh --preflight
bash scripts/run_pose_optimization_pipeline.sh --dry-run

tmux new-session -d -s poseopt \
  'cd /root/workspace/poseguard-nerf && bash scripts/run_pose_optimization_pipeline.sh'
tmux attach -t poseopt
```

按 `Ctrl-b` 再按 `d` 离开 tmux，任务继续运行。默认先执行 naive 的 GT、T0.5/1/2/5%、R0.5/1/2/5°，生成 naive 失败分析，再执行相同顺序的 CamP-style，最后生成三方法对比。共 18 次独立训练，每次 30000 步。固定相机的 9 项历史结果只读复用，全部新产物写入 `results/poseopt_sv100/`。历史单项耗时约 3 小时，完整实验约两天。

入口共享旧 SV100 流水线的文件锁，同一时刻只能有一个入口运行。另有 `ns-train` 正在运行时，预检查会停止。不要直接调用 Python 编排器绕过 shell 的锁。

查看进度：

```bash
cat results/poseopt_sv100/state.json
tail -f results/poseopt_sv100/pipeline.log
# 当前训练的详细进度与相机诊断：
tail -f results/poseopt_sv100/experiments/naive/SV100-GT/attempt_001/train.log
```

`pipeline.log` 记录阶段、命令和错误；训练/评估的完整输出在每个 attempt 下。训练过程还写 TensorBoard events；可在单独终端运行 `tensorboard --logdir results/poseopt_sv100/experiments`。

## 停止、恢复和失败补跑

在 tmux 内按 `Ctrl-c` 可停止编排器及当前 worker，状态记为 `INTERRUPTED`。终端断开无需恢复；机器重启或进程停止后需要重启入口。以下命令在旧会话结束后运行：

```bash
tmux new-session -d -s poseopt-resume \
  'cd /root/workspace/poseguard-nerf && bash scripts/run_pose_optimization_pipeline.sh --resume --retry-failed'
```

- `--resume` 校验代码、配置、数据、基线和关键依赖的指纹，跳过已验证的成功项，恢复中断项。已明确失败的项保留失败状态。
- `--resume --retry-failed` 还会补跑失败项。训练失败从最高有效 checkpoint 创建新 attempt；评估/分析失败则复用完整训练 checkpoint，重试对应阶段。
- checkpoint 恢复包括 NeRF、相机 latent、固定预条件矩阵、优化器和调度器。编排器修正 Nerfstudio 1.1.5 的追加步数行为，使最终 checkpoint 为 `step-000029999.ckpt`。
- 没有有效 checkpoint 的训练重新开始；旧 attempt、日志及失败堆栈保留。单项失败不会阻止后续独立实验；整个任务存在失败项时退出码为 1，全部成功为 0。
- 中断后随机采样序列会重新播种，Nerfstudio 原生 checkpoint 不保存完整 RNG 状态；恢复保证参数和总训练步数，不保证与不间断训练逐位一致。
- 正式运行中修改代码或参数后，应选择新的 `--run-dir`，避免把不同实现混入同一次实验。

## 输出与分析

- `manifest.json`、`environment.txt`：输入和代码指纹、环境记录。
- `state.json`：当前状态、每项阶段、所有 attempt、配置和 checkpoint 路径。
- `summary.csv`、`report.md`：fixed/naive/CamP-style 的图像指标、原坐标和 Sim(3) 对齐后的位姿误差、失败记录。
- `naive_failure_analysis.md`：进入 CamP 阶段前的 naive 阶段报告。
- `figures/`：损失、验证 PSNR、位姿误差、梯度和更新幅度曲线，以及测试视角 0/50/100 的 RGB/depth/accumulation 拼图。
- `experiments/<method>/<case>/attempt_NNN/`：独立日志、完整模型配置、checkpoint、`metrics.json`、`diagnostics.jsonl`、`optimized_poses.json`、`conditioning.json` 和测试渲染。

验证曲线使用抽样像素 batch 的 fine PSNR，因此会有采样噪声；最终 `ns-eval` 使用完整 test split 的 `fine_psnr/fine_ssim/fine_lpips`。梯度范数属于 latent 相机参数，两方法的参数空间不同；物理位姿更新幅度可用于辅助比较。depth 拼图统一以 near=2、far=6 映射灰度；原始 float 数组同时保存为 `.npy`。

图像评估始终使用原始验证/测试相机，Sim(3) 对齐只用于位姿诊断，不调整测试相机或图像指标。原坐标误差反映绝对恢复与全局漂移，对齐误差辅助识别相对位姿是否改善。自动分析标记退化、停滞、振荡及图像/位姿指标不一致；原因解释只是待验证假设。单种子实验不提供统计显著性保证。

## 方法定义

项目内扩展通过 `NERFSTUDIO_METHOD_CONFIGS` 注册 `poseguard-naive`、`poseguard-camp`，不修改安装的 Nerfstudio。两方法从相同场景网络初始化、相同扰动数据和零相机修正开始。场景优化仍为 stock vanilla-NeRF 的 RAdam（lr=5e-4），采样、网络与背景保持原配置；诊断采样保存并恢复 RNG，避免额外影响训练随机流。

每个训练相机学习六维 world-frame `SO3 × R3` 修正，前三维为平移，后三维为 axis-angle：

```text
R' = Exp(ω) R
 t' = t + Δt
```

训练时在场景采样前更新 ray origin 与 direction，验证/测试不应用相机表。两方法的相机优化器均为 Adam（lr 从 1e-3 指数衰减至 1e-4），从第零步开始优化；无位姿正则、warmup、频率调度或内参优化。GT 只在模型和预条件构建完成后用于诊断。

CamP-style 使用投影代理问题的完整六维 ZCA 白化，参考 [CamP 论文第 5 节](https://arxiv.org/html/2308.10902v2)：

```text
Σ = JᵀJ / N
P = (Σ + 0.1 diag(Σ) + 1e-8 I)^(-1/2)
δ = P z
```

每台相机在初始视锥中采样 4096 个点，像素均匀采样，归一化 ray distance 在 [2,6] 上均匀采样；投影输出为 `(u-cx)/width, (v-cy)/height`，使用 OpenGL 坐标轴。解析 Jacobian 在初始位姿处计算，float64 求解矩阵后转 float32，固定存为 checkpoint buffer。采样使用隔离的 seed=42 RNG。与原论文使用 Zip-NeRF、更多相机参数和其深度采样分布不同，本实现是控制变量的核心机制实验，不是官方完整复现。

## 快速验证

```bash
PYTHONPATH=. /root/miniconda3/bin/python -m unittest discover -s tests -v
bash scripts/run_pose_optimization_pipeline.sh \
  --smoke --run-dir results/poseopt_smoke_new --steps 4 --interval 2 --cases SV100-R1
```

smoke 降低分辨率、ray batch、场景采样数量和代理点数，仅用于检查端到端链路，结果不能与正式实验比较。smoke 与正式 run-dir 强制隔离。自定义实验可用 `--cases`、`--steps`、`--interval`、`--run-dir`；恢复时必须使用同一组参数。
