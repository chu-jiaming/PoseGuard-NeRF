# GT-condition alignment diagnostic

Alignment is fitted using GT and optimized TRAIN camera centers only. No test-image fitting, no retraining.
The same transform rotates test orientations and scales/translates test centers. Sampling bounds scale consistently.
A center-fit Sim(3) cannot remove local pose errors; residual degradation does not identify its cause.

Resolution scale: 0.25; test views: 20. Single scene and training seed.
Raw metrics source: paired rerender.

| Method | Raw PSNR | Aligned PSNR | Delta | Raw SSIM | Aligned SSIM | Raw LPIPS | Aligned LPIPS | Scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 28.306 | 28.306 | +0.000 | 0.9430 | 0.9430 | 0.0478 | 0.0478 | 1.00000 |
| naive | 18.929 | 25.786 | +6.857 | 0.7275 | 0.9115 | 0.0797 | 0.0499 | 1.01081 |
| camp | 21.356 | 28.598 | +7.242 | 0.8073 | 0.9428 | 0.0631 | 0.0397 | 1.01028 |
| naive_warmup | 24.994 | 29.029 | +4.035 | 0.8902 | 0.9472 | 0.0514 | 0.0393 | 1.00370 |
| camp_warmup | 22.970 | 28.488 | +5.518 | 0.8492 | 0.9426 | 0.0554 | 0.0384 | 1.00833 |
