# PoseGuard-NeRF delivery

找到 45 个run；缺失 0 个。核心15个PSNR及两个参考checkpoint哈希已通过。
P0/P1：96张800×800渲染、96张误差图、16张GT；视角0、50、100、175（175展示斜侧机械细节）。
P2：4段200帧400×400视频、4个GIF和带标签的T5并排MP4。P3标量与36份diagnostics；P4见checkpoints_index.json。
raw均值检查最大偏差为 3.6 dB（精确 3.617626826 dB）。raw检查主要只做均值级比对，没有原始raw全分辨率逐图记录可供核对；例外是GT/fixed另用历史恒等对齐逐图指标核对过4张图，偏差0。不能把均值检查当成0.05 dB逐图验证。
四种aligned方法的16张800×800图严格核对历史逐图指标，最大偏差0 dB；400×400指标单独记录，不与800×800比较。
raw实际near=0、far=6；aligned调用原transform_poses，实际near=0、far=6×scale，reset_near_plane=true。白背景，rgb_fine。误差图为未量化RGB每像素L2，turbo [0,0.3]。
目录差异：outputs及poseopt_sv100、部分warmup attempt通过软链接指向/root/autodl-tmp；fixed/T1为lego_tnoise_1pct；aligned progress实际在results/alignment_gt_full。
原代码、实验输出、checkpoint、数据及event文件未修改；没有新训练。
逐帧PNG未超过300MB，已纳入压缩包。
各关耗时和验收见gates/A.json、B.json及C.json；关卡不会自动衔接。
包路径、大小、SHA256见包外gates/C.md及archive_receipt.json；哈希无法自包含于其所校验的压缩包。

## CSV标签行数

| tag | rows |
|---|---:|
| ETA (time) | 1350000 |
| Eval Images Metrics Dict (all images)/coarse_psnr | 45 |
| Eval Images Metrics Dict (all images)/fine_lpips | 45 |
| Eval Images Metrics Dict (all images)/fine_psnr | 45 |
| Eval Images Metrics Dict (all images)/fine_ssim | 45 |
| Eval Images Metrics Dict (all images)/fps | 45 |
| Eval Images Metrics Dict (all images)/num_rays_per_sec | 45 |
| Eval Images Metrics Dict (all images)/psnr | 45 |
| Eval Images Metrics/coarse_psnr | 2655 |
| Eval Images Metrics/fine_lpips | 2655 |
| Eval Images Metrics/fine_psnr | 2655 |
| Eval Images Metrics/fine_ssim | 2655 |
| Eval Images Metrics/num_rays | 2655 |
| Eval Images Metrics/psnr | 2655 |
| Eval Loss | 2655 |
| Eval Loss Dict/rgb_loss_coarse | 2655 |
| Eval Loss Dict/rgb_loss_fine | 2655 |
| GPU Memory (MB) | 135000 |
| Test Rays / Sec | 2655 |
| Train Iter (time) | 1350000 |
| Train Loss | 135000 |
| Train Loss Dict/rgb_loss_coarse | 135000 |
| Train Loss Dict/rgb_loss_fine | 135000 |
| Train Rays / Sec | 1349910 |
| Train Total (time) | 45 |
| learning_rate/camera_opt | 1080000 |

## P0/P1 PSNR

| condition | method | mode | view | PSNR |
|---|---|---|---:|---:|
| SV100-GT | fixed | raw | 0 | 27.223795 |
| SV100-GT | fixed | raw | 50 | 28.454309 |
| SV100-GT | fixed | raw | 100 | 28.452311 |
| SV100-GT | fixed | raw | 175 | 26.635462 |
| SV100-GT | naive | raw | 0 | 17.735168 |
| SV100-GT | naive | raw | 50 | 20.259108 |
| SV100-GT | naive | raw | 100 | 18.841879 |
| SV100-GT | naive | raw | 175 | 16.385822 |
| SV100-GT | naive | aligned | 0 | 24.344198 |
| SV100-GT | naive | aligned | 50 | 25.700920 |
| SV100-GT | naive | aligned | 100 | 24.643089 |
| SV100-GT | naive | aligned | 175 | 22.571732 |
| SV100-GT | camp | raw | 0 | 20.480253 |
| SV100-GT | camp | raw | 50 | 21.083437 |
| SV100-GT | camp | raw | 100 | 21.441933 |
| SV100-GT | camp | raw | 175 | 17.739981 |
| SV100-GT | camp | aligned | 0 | 25.650791 |
| SV100-GT | camp | aligned | 50 | 27.302214 |
| SV100-GT | camp | aligned | 100 | 27.523165 |
| SV100-GT | camp | aligned | 175 | 25.491484 |
| SV100-GT | naive_warmup | raw | 0 | 22.124289 |
| SV100-GT | naive_warmup | raw | 50 | 28.044697 |
| SV100-GT | naive_warmup | raw | 100 | 25.289804 |
| SV100-GT | naive_warmup | raw | 175 | 21.499994 |
| SV100-GT | naive_warmup | aligned | 0 | 26.497204 |
| SV100-GT | naive_warmup | aligned | 50 | 28.129665 |
| SV100-GT | naive_warmup | aligned | 100 | 27.915257 |
| SV100-GT | naive_warmup | aligned | 175 | 25.803513 |
| SV100-GT | camp_warmup | raw | 0 | 24.318935 |
| SV100-GT | camp_warmup | raw | 50 | 23.411909 |
| SV100-GT | camp_warmup | raw | 100 | 23.329647 |
| SV100-GT | camp_warmup | raw | 175 | 19.434631 |
| SV100-GT | camp_warmup | aligned | 0 | 26.323505 |
| SV100-GT | camp_warmup | aligned | 50 | 28.027498 |
| SV100-GT | camp_warmup | aligned | 100 | 27.821537 |
| SV100-GT | camp_warmup | aligned | 175 | 24.991163 |
| SV100-T5 | fixed | raw | 0 | 14.531160 |
| SV100-T5 | fixed | raw | 50 | 15.424973 |
| SV100-T5 | fixed | raw | 100 | 13.503274 |
| SV100-T5 | fixed | raw | 175 | 13.722331 |
| SV100-T5 | naive | raw | 0 | 19.539042 |
| SV100-T5 | naive | raw | 50 | 20.405630 |
| SV100-T5 | naive | raw | 100 | 18.844475 |
| SV100-T5 | naive | raw | 175 | 17.436981 |
| SV100-T5 | camp | raw | 0 | 17.767853 |
| SV100-T5 | camp | raw | 50 | 22.052055 |
| SV100-T5 | camp | raw | 100 | 20.516823 |
| SV100-T5 | camp | raw | 175 | 20.106083 |
| SV100-T5 | naive_warmup | raw | 0 | 25.462740 |
| SV100-T5 | naive_warmup | raw | 50 | 25.092705 |
| SV100-T5 | naive_warmup | raw | 100 | 25.244631 |
| SV100-T5 | naive_warmup | raw | 175 | 22.168625 |
| SV100-T5 | camp_warmup | raw | 0 | 22.368828 |
| SV100-T5 | camp_warmup | raw | 50 | 21.705992 |
| SV100-T5 | camp_warmup | raw | 100 | 21.443264 |
| SV100-T5 | camp_warmup | raw | 175 | 17.540792 |
| SV100-R5 | fixed | raw | 0 | 12.795829 |
| SV100-R5 | fixed | raw | 50 | 13.024708 |
| SV100-R5 | fixed | raw | 100 | 12.221133 |
| SV100-R5 | fixed | raw | 175 | 13.117497 |
| SV100-R5 | naive | raw | 0 | 17.253752 |
| SV100-R5 | naive | raw | 50 | 22.198116 |
| SV100-R5 | naive | raw | 100 | 18.877028 |
| SV100-R5 | naive | raw | 175 | 15.778337 |
| SV100-R5 | camp | raw | 0 | 15.722897 |
| SV100-R5 | camp | raw | 50 | 17.246365 |
| SV100-R5 | camp | raw | 100 | 15.959490 |
| SV100-R5 | camp | raw | 175 | 13.488349 |
| SV100-R5 | naive_warmup | raw | 0 | 16.682283 |
| SV100-R5 | naive_warmup | raw | 50 | 18.993124 |
| SV100-R5 | naive_warmup | raw | 100 | 19.988968 |
| SV100-R5 | naive_warmup | raw | 175 | 16.819361 |
| SV100-R5 | camp_warmup | raw | 0 | 14.963828 |
| SV100-R5 | camp_warmup | raw | 50 | 17.226337 |
| SV100-R5 | camp_warmup | raw | 100 | 17.961039 |
| SV100-R5 | camp_warmup | raw | 175 | 14.719579 |
| SV100-T1 | fixed | raw | 0 | 20.045074 |
| SV100-T1 | fixed | raw | 50 | 22.311666 |
| SV100-T1 | fixed | raw | 100 | 21.249514 |
| SV100-T1 | fixed | raw | 175 | 20.195431 |
| SV100-T1 | naive | raw | 0 | 22.063765 |
| SV100-T1 | naive | raw | 50 | 18.897251 |
| SV100-T1 | naive | raw | 100 | 17.592247 |
| SV100-T1 | naive | raw | 175 | 14.783033 |
| SV100-T1 | camp | raw | 0 | 17.207783 |
| SV100-T1 | camp | raw | 50 | 18.507387 |
| SV100-T1 | camp | raw | 100 | 17.706535 |
| SV100-T1 | camp | raw | 175 | 14.673221 |
| SV100-T1 | naive_warmup | raw | 0 | 22.561476 |
| SV100-T1 | naive_warmup | raw | 50 | 25.099747 |
| SV100-T1 | naive_warmup | raw | 100 | 24.968843 |
| SV100-T1 | naive_warmup | raw | 175 | 23.785057 |
| SV100-T1 | camp_warmup | raw | 0 | 20.870274 |
| SV100-T1 | camp_warmup | raw | 50 | 22.538006 |
| SV100-T1 | camp_warmup | raw | 100 | 23.000193 |
| SV100-T1 | camp_warmup | raw | 175 | 19.509480 |
