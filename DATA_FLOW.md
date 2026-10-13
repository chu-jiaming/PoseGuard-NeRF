# 数据与产物流

## 输入

`POSEGUARD_DATA` 指向 NeRF-Synthetic 根目录，默认是仓库同级 `data/nerf_synthetic`。Lego 含 100 个训练、100 个验证、200 个测试视角。`transforms_*.json` 指定图片与 camera-to-world 外参，`camera_angle_x` 指定水平视场角。

`prepare_datasets.py` 调用 `perturb_poses.py` 生成八种噪声数据：T0.5/1/2/5%、R0.5/1/2/5°。只改变训练外参，图片通过软链接复用；保存 GT 副本与噪声元数据。平移比例以训练相机到轨迹中心的平均半径为基准。

本机另存的稀疏视角输入供后续项目使用，不参与这里的 45 次训练。

## 训练与评估

图片与相机 → Blender parser → 初始 rays → Fixed 或 SO3×R3 相机修正 → coarse/fine NeRF → RGB MSE → 更新参数。

9 次 Fixed + 18 次无 warm-up 联合优化 + 18 次 warm-up 联合优化，共 45 次训练。Warm-up 不重复 Fixed。全部为 Lego、seed 42、30,000 步，checkpoint 从零计数至 29,999。

训练过程抽样验证像素用于监测，最终图像指标用全部测试图像。主训练流水线保留原始测试坐标；独立对齐诊断用训练相机中心拟合 GT→重建坐标的 Sim(3)，统一变换测试相机和采样距离，不使用测试图像拟合。

位姿审计覆盖 36 组优化结果；对齐渲染仅覆盖 GT 的五种设置。位姿平移百分比按 GT 相机到世界原点的平均距离归一化，与噪声标签的轨迹中心半径不同。

## 运行产物与发布档案

| 路径 | 用途 | Git |
|---|---|---|
| `outputs/` | Fixed 配置、checkpoint、TensorBoard | 忽略 |
| `logs/` | 运行日志 | 忽略 |
| `results/poseopt_sv100/` | 无 warm-up 训练、指标和报告 | 忽略 |
| `results/poseopt_sv100_warmup2500/` | warm-up 训练与配对分析 | 忽略 |
| `results/alignment_gt_{preview,full}/` | 两种分辨率的对齐渲染与报告 | 忽略 |
| `results/alignment_pose_audit/` | 位姿审计 | 忽略 |
| `artifacts/` | 上述产物的小型发布快照 | 纳入 |

每个训练 attempt 包含配置、权重、优化位姿、图像指标、每 1000 步诊断、完成标记和日志。状态与 manifest 记录阶段、来源、输入和代码指纹。恢复规则见 [POSE_OPTIMIZATION.md](POSE_OPTIMIZATION.md)。

本机部分原始产物已迁到 `/root/autodl-tmp/poseguard-nerf`，工作区用软链接访问。发布档案使用普通文件，不依赖该数据盘；原实验未移动或改写。原已跟踪的运行记录归档到 `artifacts/historical/`，运行目录从 Git 索引移除，但磁盘数据保留。

[档案清单与说明](artifacts/README.md) · [新机器复现](REPRODUCING.md) · [最终项目结论](README.md)
