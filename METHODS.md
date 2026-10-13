# Methods: Naive and CamP-style Pose Optimization

本文第 1–5 节描述首轮无 warm-up 实验的方法。后续 warm-up 只改变相机学习率日程，见 [WARMUP_EXPERIMENTS.md](WARMUP_EXPERIMENTS.md)；测试渲染对齐协议见 [ALIGNMENT_DIAGNOSTIC.md](ALIGNMENT_DIAGNOSTIC.md)。两种方法共享 Nerfstudio 1.1.5 的 vanilla-NeRF、六维位姿参数化和图像重建损失；核心差异是每台相机的固定预条件矩阵。

本仓库实现的是受 [CamP: Camera Preconditioning for Neural Radiance Fields](https://arxiv.org/abs/2308.10902) 启发的外参优化实验。名称使用 **CamP-style**，以区分本实现和官方完整实验设置。

## 1. 共享问题与坐标约定

训练输入为图片 $I_i$、固定内参 $K_i$ 和初始 camera-to-world 外参 $T_i^0=[R_i^0\mid t_i^0]$。场景参数记为 $\theta$，训练相机数为 $M$。

每台相机的物理修正为六维列向量：

$$
\delta_i=\begin{bmatrix}\Delta t_i\\\omega_i\end{bmatrix}\in\mathbb R^6.
$$

前三维是世界坐标中的平移修正，后三维是世界坐标中的 axis-angle 旋转修正，角度单位为弧度。相机更新为：

$$
R_i=\operatorname{Exp}([\omega_i]_\times)R_i^0,
\qquad t_i=t_i^0+\Delta t_i.
$$

这是 SO(3) × R³ 更新：旋转修正左乘初始朝向，平移独立叠加到初始相机位置。旋转映射复用 Nerfstudio 的 `exp_map_SO3xR3`。内参保持固定。

对由初始相机生成的训练 ray $(o_{ij}^0,d_{ij}^0)$，更新为：

$$
o_{ij}=o_{ij}^0+\Delta t_i,
\qquad d_{ij}=\operatorname{Exp}([\omega_i]_\times)d_{ij}^0.
$$

更新发生在 NeRF 沿 ray 采样之前，仅在模型的训练分支应用。两种方法从相同场景初始化、相同训练数据和零位姿修正开始。

## 2. Naive joint optimization

每台相机直接学习 $z_i\in\mathbb R^6$，定义：

$$
P_i=I_6,\qquad \delta_i=P_i z_i=z_i,\qquad z_i^0=0.
$$

对训练 batch $\mathcal B$，由当前场景和修正相机渲染 coarse/fine RGB，并联合优化 $\theta$ 和所有 $z_i$：

$$
\min_{\theta,\{z_i\}}\mathcal L_{\mathrm{RGB}},
$$

$$
\mathcal L_{\mathrm{RGB}}=
\frac{1}{3|\mathcal B|}
\sum_{(i,j)\in\mathcal B}
\left(
\|\hat C^{\mathrm{coarse}}_{ij}-C_{ij}\|_2^2+
\|\hat C^{\mathrm{fine}}_{ij}-C_{ij}\|_2^2
\right).
$$

RGB 使用 Nerfstudio 的白色背景合成规则；coarse/fine 两项的权重均为 1。位姿没有额外监督损失或正则项，所有训练相机都可更新，没有固定锚点相机。

```text
initial training rays
    → delta_i = z_i
    → corrected rays
    → coarse/fine NeRF rendering
    → RGB reconstruction loss
    → update scene theta and camera z_i
```

这里的 naive 是本仓库的无正则 SO3×R3 联合优化设置。Nerfstudio 原生提供 `CameraOptimizer`，nerfacto 已接入该组件；本仓库给 vanilla-NeRF 增加接入，并自行持有六维参数。现有实验不能直接代表采用原生默认正则的 nerfacto 相机优化效果。

## 3. CamP-style joint optimization

CamP-style 保留上述相机更新、场景模型、损失和优化器，改为学习 latent $z_i$：

$$
\delta_i=P_i z_i,\qquad z_i^0=0.
$$

$P_i\in\mathbb R^{6\times6}$ 是每台相机独立的、初始化时计算并固定的完整矩阵。

### 3.1 代理空间点

每台相机在初始视锥中采样 $N=4096$ 个点。像素坐标均匀采样：

$$
u\sim\mathcal U(0,W_i),\quad v\sim\mathcal U(0,H_i).
$$

采用 OpenGL 相机坐标约定，前方为负 z。初始单位方向为：

$$
a=\operatorname{normalize}\left(
\begin{bmatrix}(u-c_{x,i})/f_{x,i}\\-(v-c_{y,i})/f_{y,i}\\-1\end{bmatrix}
\right).
$$

沿单位 ray 的距离 $s\sim\mathcal U(2,6)$，生成世界坐标点：

$$
X_{i\ell}=R_i^0(sa)+t_i^0.
$$

因此采样的是沿 ray 的距离，而非相机 z 轴深度。范围与本次 vanilla-NeRF 的 near/far 配置一致。采样采用 seed=42 的隔离 CPU generator，按相机顺序依次取样，不消耗场景训练的 RNG。生成后，这些世界点在求 Jacobian 时视为常量。

### 3.2 投影 Jacobian

对固定世界点 $X$，当前相机坐标为：

$$
q_i(\delta_i)=R_i(\delta_i)^\top[X-t_i(\delta_i)].
$$

投影代理输出使用归一化、去中心的像素坐标：

$$
\pi_i(X;\delta_i)=
\begin{bmatrix}
(f_{x,i}/W_i)q_x/(-q_z)\\
-(f_{y,i}/H_i)q_y/(-q_z)
\end{bmatrix}
=\begin{bmatrix}(u-c_{x,i})/W_i\\(v-c_{y,i})/H_i\end{bmatrix}.
$$

在 $\delta_i=0$ 处，拼接所有点的解析导数：

$$
J_i=\left.
\frac{\partial[\pi_i(X_{i1});\ldots;\pi_i(X_{iN})]}{\partial\delta_i}
\right|_{\delta_i=0}
\in\mathbb R^{2N\times6}.
$$

代码中的三维中间导数为：

$$
\left.\frac{\partial q_i}{\partial\delta_i}\right|_0
=\left[-(R_i^0)^\top\;\middle|\;(R_i^0)^\top[X-t_i^0]_\times\right].
$$

`projection_jacobian()` 实现该解析 Jacobian；测试使用有限差分核对坐标、符号和参数顺序。

### 3.3 带阻尼的 ZCA 变换

构造按点数平均的投影敏感度矩阵：

$$
\Sigma_i=\frac{1}{N}J_i^\top J_i.
$$

保留全部非对角项，用于表示旋转和平移参数影响之间的相关性。加入阻尼：

$$
A_i=\Sigma_i+0.1\operatorname{diag}(\Sigma_i)+10^{-8}I_6,
$$

其中 $\operatorname{diag}(\Sigma_i)$ 表示只保留原矩阵对角元素的对角矩阵。对称特征分解 $A_i=Q_i\Lambda_iQ_i^\top$ 后：

$$
P_i=A_i^{-1/2}=Q_i\Lambda_i^{-1/2}Q_i^\top.
$$

矩阵计算使用 CPU float64，完成后转为 float32，作为模型 buffer 保存到 checkpoint。训练过程中保持固定。

论文中的 inverse preconditioner 在本仓库代码里命名为 `pose_preconditioner`，本文用 $P_i$ 表示。该实现按点数平均并使用上述归一化投影，矩阵的绝对尺度由这些约定决定。

没有阻尼且矩阵满秩时，理想变换满足 $P_i^\top\Sigma_i P_i=I$；实际带阻尼实现提供近似的尺度归一化和去相关。它基于投影代理问题，不保证真实 NeRF 损失收敛或测试指标改善。

### 3.4 联合训练

每次前向计算通过 $\delta_i=P_i z_i$ 修正 rays，使用与 naive 相同的 RGB 损失更新 $\theta$ 和 $z_i$。反向传播自然给出：

$$
\nabla_{z_i}\mathcal L=P_i^\top\nabla_{\delta_i}\mathcal L.
$$

Adam 作用于 latent $z_i$。这改变了 Adam 所处的参数空间；相同的 latent 学习率不意味着两种方法有相同的物理位姿步长。

```text
initial cameras
    → sample proxy points
    → projection Jacobian J_i
    → damped sensitivity matrix A_i
    → fixed P_i = A_i^(-1/2)

each training iteration:
    z_i → delta_i = P_i z_i → corrected rays → RGB loss
        → update scene theta and latent z_i
```

## 4. 共享训练设置

| 设置 | 正式实验取值 |
|---|---|
| 场景模型 | Nerfstudio 1.1.5 vanilla-NeRF，coarse/fine 两个 field |
| 初始场景与相机修正 | 从头初始化；修正为零 |
| 场景优化器 | RAdam，lr=5e-4，eps=1e-8，无学习率调度 |
| 相机优化器 | Adam，初始 lr=1e-3，eps=1e-8 |
| 相机学习率 | 在 30000 步内指数衰减到 1e-4，warmup_steps=0 |
| 训练 ray batch | 1024 |
| Coarse samples | 64 |
| Importance samples 参数 | 128 |
| Near / far | 2 / 6 |
| 背景 / 数据尺度 | 白色 / Blender scale_factor=1 |
| Mixed precision | 关闭 |
| 随机种子 / 训练步数 | 42 / 30000 |
| 位姿正则、GT 位姿监督、固定锚点 | 均未启用 |
| 频率 coarse-to-fine 调度、内参优化 | 均未启用 |

相机学习率的名义调度为 $\eta(k)=10^{-3}(10^{-1})^{\min(k/30000,1)}$。两种方法采用相同 latent Adam 设置，CamP-style 的物理步长还受到 $P_i$ 影响。

## 5. 数据与评估

使用 NeRF-Synthetic Lego：训练 100 帧、验证 100 帧、测试 200 帧。9 个条件为 GT、T0.5/1/2/5%、R0.5/1/2/5°；图片与内参不变，仅扰动训练外参。平移比例以生成元数据记录的轨迹中心平均半径为基准，旋转幅度单位为度。

正确训练位姿在模型与预条件构建后才读取，用于诊断。验证、测试使用原始正确相机，不应用训练修正表，也不进一步优化测试相机。

最终图像评估报告 `fine_psnr`、`fine_ssim`、`fine_lpips`。相机诊断报告原坐标误差，以及用训练相机中心估计 Umeyama Sim(3) 后的对齐误差；该变换只用于位姿诊断，不调整测试相机或场景模型。

`translation_radius_pct_mean` 按 GT 相机到世界原点的平均距离归一化，和噪声标签采用的轨迹中心半径不同。训练曲线中的验证 PSNR 来自采样 batch，最终指标来自完整测试集。

## 6. 已完成结果与解释边界

27 项正式运行已完成，包括 9 项 fixed 基线和 18 项位姿优化。CamP-style 在 6/9 个条件下的测试 PSNR 高于 naive；T5% 和 R2° 有明显改善，GT、T0.5% 等条件出现退化。GT 条件的测试 PSNR 为 fixed 27.12、naive 17.91、CamP-style 19.90 dB。

这些结果属于单场景、单种子、无位姿约束的联合优化实验。首轮测试图像没有做坐标系对齐。后续已完成 18 次 warm-up 训练及 GT 条件的完整渲染对齐诊断，确认原坐标指标对全局漂移敏感；残余差距不能单凭此实验归因。最新结果见 README。不能据此判断官方 CamP 或原生 nerfacto 的效果。

结果与数据流：

- [27 项指标汇总](artifacts/historical/results/poseopt_sv100/summary.csv)
- [完整分析报告](artifacts/historical/results/poseopt_sv100/report.md.txt)
- [数据流与产物说明](DATA_FLOW.md)
- [运行与恢复说明](POSE_OPTIMIZATION.md)

## 7. 实现位置

| 代码 | 对应方法步骤 |
|---|---|
| [`poseguard/geometry.py`](poseguard/geometry.py) | 位姿更新、代理投影、解析 Jacobian、矩阵预条件、Sim(3) 诊断 |
| [`poseguard/methods.py`](poseguard/methods.py) | 参数注册、训练 ray 更新、naive/CamP 方法配置 |
| [`poseguard/worker.py`](poseguard/worker.py) | 随机种子、训练、恢复、诊断与位姿/渲染导出 |
| [`tests/test_poseguard.py`](tests/test_poseguard.py) | 几何、有限差分、梯度、评估隔离、buffer 保存恢复 |

## Reference

Keunhong Park, Philipp Henzler, Ben Mildenhall, Jonathan T. Barron, and Ricardo Martin-Brualla. **CamP: Camera Preconditioning for Neural Radiance Fields.** SIGGRAPH Asia 2023. [Paper](https://arxiv.org/abs/2308.10902) · [Project](https://camp-nerf.github.io/).
