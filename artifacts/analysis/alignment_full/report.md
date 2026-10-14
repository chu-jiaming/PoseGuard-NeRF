# GT-condition alignment diagnostic

Alignment is fitted using GT and optimized TRAIN camera centers only. No test-image fitting, no retraining.
The same transform rotates test orientations and scales/translates test centers. Sampling bounds scale consistently.
A center-fit Sim(3) cannot remove local pose errors; residual degradation does not identify its cause.

Resolution scale: 1.0; test views: 200. Single scene and training seed.
Raw metrics source: existing full-resolution 200-view metrics.

| Method | Raw PSNR | Aligned PSNR | Delta | Raw SSIM | Aligned SSIM | Raw LPIPS | Aligned LPIPS | Scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 27.117 | 27.117 | +0.000 | 0.9003 | 0.9003 | 0.0790 | 0.0790 | 1.00000 |
| naive | 17.909 | 23.386 | +5.478 | 0.7508 | 0.8384 | 0.1486 | 0.1027 | 1.01081 |
| camp | 19.900 | 26.037 | +6.137 | 0.7829 | 0.8794 | 0.1236 | 0.0937 | 1.01028 |
| naive_warmup | 22.707 | 26.419 | +3.711 | 0.8240 | 0.8854 | 0.1052 | 0.0938 | 1.00370 |
| camp_warmup | 21.076 | 25.979 | +4.904 | 0.7965 | 0.8810 | 0.1117 | 0.0928 | 1.00833 |
