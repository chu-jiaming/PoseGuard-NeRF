# 当前数据流与实验流程

本工作区保留当前正式实验及其输入：9 项固定相机基线、9 项 naive、9 项 CamP-style。位姿优化流水线已经全部成功完成。旧验证产物、废弃脚本和迁移前 Git 历史已清理。

## 1. 输入：图片与相机参数

数据根目录是 `/root/workspace/data/`，位于本 Git 仓库之外。

```text
data/
├── nerf_synthetic.zip                原始下载压缩包
├── nerf_synthetic/
│   ├── lego/                        当前实验的原始数据与正确位姿
│   ├── chair/、drums/、ficus/        其他原始场景
│   ├── hotdog/、materials/、mic/、ship/
│   ├── lego_sv100_tnoise_*pct/       0.5%、2%、5% 平移扰动
│   ├── lego_tnoise_1pct/             1% 平移扰动，历史命名保留以维持实验引用
│   └── lego_sv100_rnoise_*deg/       0.5°、1°、2°、5° 旋转扰动
├── nerf_synthetic_sparse/
│   └── lego_sv{50,20,10,5,3}/        稀疏视角输入，供后续学习使用
├── parser.py                        稀疏视角数据生成脚本
└── README.md                        输入数据说明
```

一个原始场景包含 `train/`、`val/`、`test/` 图片及 `transforms_train/val/test.json`。Lego 的三个 split 分别有 100、100、200 帧。

每个 JSON 的 `frames` 列表指定使用哪些图片；`file_path` 指向图片，`transform_matrix` 是 4×4 camera-to-world 外参，`camera_angle_x` 提供水平视场角。Blender parser 根据视场角与图片尺寸计算焦距。

因此，一个训练样本由“图片像素 + 内参 + 对应外参”构成，目录里有图片并不意味着每张都参与训练：参与训练的列表由 `transforms_train.json` 决定。

## 2. 控制变量：生成扰动数据或减少视角

位姿扰动由 `scripts/perturb_poses.py` 生成。每个扰动目录包含：

```text
transforms_train.json                改动后的训练外参
transforms_train_gt.json             正确训练外参的参考副本
transforms_val.json                  正确验证外参
transforms_test.json                 正确测试外参
pose_noise_metadata.json             噪声类型、幅度、种子和逐帧扰动记录
train/、val/、test/                  指向原始 Lego 图片的软链接
```

扰动实验保持图片、100 个训练视角和内参一致，只改变训练外参。平移实验保持旋转不变；旋转实验保持平移不变。验证、测试 split 与原始 Lego 一致。

`T2` 的噪声大小是元数据 `camera_radius` 的 2%；该 radius 是训练相机到轨迹平均中心的平均距离。`R2` 表示每台相机施加 2° 的旋转扰动。文件路径保留原命名，因为正式配置与指纹引用这些路径。

稀疏视角分支由 `data/parser.py` 生成：在正确训练相机列表中均匀选取 50/20/10/5/3 帧，验证和测试保持不变，图片仍通过软链接复用。当前 27 项正式对比使用 100 个训练视角，未使用稀疏分支。

## 3. 训练：同一数据条件，比较三种方法

```text
图片像素 + camera_angle_x + transforms_train.json
    → Blender parser 生成相机
    → 像素对应的 ray origin / direction
    → fixed / naive / CamP-style
    → 沿 ray 采样、NeRF 预测、体渲染
    → 与训练图片比较损失
    → 更新场景网络，naive/CamP 同时更新训练相机修正
    → 保存 config、checkpoint、TensorBoard 记录
```

- fixed：相机保持输入值，训练 vanilla-NeRF。
- naive：直接优化每个训练相机的六维 SO3×R3 修正。
- CamP-style：优化 latent 参数，通过固定投影 Jacobian 预条件矩阵转换为物理位姿修正。

三种方法覆盖 GT、T0.5/1/2/5%、R0.5/1/2/5°，每项 30000 步。GT 条件指初始相机正确；相机优化方法仍可改变它们。正确训练位姿只用于优化误差诊断，不作为监督损失或预条件矩阵的输入。

固定相机入口是 `scripts/run_full_sv100_pipeline.sh`，编排器为 `scripts/sv100_pipeline.py`。位姿优化入口是 `scripts/run_pose_optimization_pipeline.sh`，编排器为 `scripts/pose_optimization_pipeline.py`，具体模型与 worker 位于 `poseguard/`。

`sv100_pipeline.py` 同时提供位姿流水线复用的数据条件和校验逻辑，因此是当前依赖。运行和恢复操作见 [POSE_OPTIMIZATION.md](POSE_OPTIMIZATION.md)。

## 4. 评估：使用未扰动的测试相机

```text
最终 checkpoint + transforms_test.json
    → 用正确测试相机渲染 200 个测试视角
    → 与测试图片比较
    → metrics.json：fine_psnr、fine_ssim、fine_lpips
```

验证 split 用于训练过程监测；最终图像指标来自测试 split。验证与测试不应用训练相机修正表。`transforms_train_gt.json`/原始 GT 训练位姿用于衡量相机恢复程度，不能替代测试图像指标。

位姿诊断同时提供原坐标和 Sim(3) 对齐后的误差。对齐只作用于诊断，不改变模型和测试图像评估。当前诊断中的 `translation_radius_pct_mean` 按 GT 相机到世界原点的平均距离归一化；它与扰动元数据采用的轨迹中心半径不同，数值不能直接等同于噪声标签百分比。

## 5. 产物：从完整运行追溯到汇总报告

固定相机产物分别存放：

```text
outputs/<实验名>/vanilla-nerf/<时间戳>/    模型配置、权重和 TensorBoard
logs/{gt,translation,rotation}/           训练、校验、评估日志
results/SV100-*/                          指标与运行元数据
results/SV100_full_summary.csv            9 项固定相机汇总
```

位姿优化产物集中存放：

```text
results/poseopt_sv100/
├── state.json                           18 项状态和各产物路径
├── manifest.json                        数据、基线、代码与关键依赖指纹
├── environment.txt                      环境记录
├── pipeline.log                         阶段与恢复过程
├── summary.csv                          27 行 fixed/naive/CamP 汇总
├── report.md                            完整结果报告
├── naive_failure_analysis.md            naive 阶段分析
├── figures/                             收敛曲线及三方法拼图
├── fixed_renders/                       用基线模型额外生成的对比渲染
└── experiments/{naive,camp}/<条件>/attempt_001/
    ├── outputs/.../config.yml
    ├── outputs/.../nerfstudio_models/step-000029999.ckpt
    ├── outputs/.../events.out.tfevents.*
    ├── train.log、eval.log、analyze.log
    ├── metrics.json                     测试图像指标
    ├── diagnostics.jsonl                损失、梯度、位姿误差与更新幅度
    ├── optimized_poses.json             优化后的训练位姿
    ├── conditioning.json                CamP 预条件诊断
    ├── train_complete.json
    ├── analysis_complete.json
    └── view_*.png / view_*.npy           视角 0/50/100 的渲染与原始数组
```

checkpoint 从零计数，`step-000029999.ckpt` 对应 30000 步。`fixed_renders/` 复用已有基线 checkpoint，只增加对比视角渲染，不重复训练基线。

最终分析把图像质量、相机恢复、收敛曲线和定性渲染结合起来，检查退化、振荡、停滞及图像与位姿指标不一致。自动标记是诊断线索，原因解释仍需要验证。

## 6. 当前版本管理

Git 历史重新建立为当前工作区的一次初始提交，旧的迁移前提交已移除。当前代码、测试、文档、正式配置、指标、状态与指纹纳入 Git。

大型 checkpoint、TensorBoard、运行日志和图片/数组保存在磁盘并由 `.gitignore` 排除。原始数据和学习纲要位于仓库之外。正式实验的代码、数据和基线指纹保持不变。
