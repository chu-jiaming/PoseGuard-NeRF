# Pose optimization — SUCCESS

Single seed 42; vanilla-NeRF; target 30000 steps. Fixed baselines: 30000 steps.
Test cameras remain in original coordinates. Sim(3) alignment is used only for pose diagnostics.

| Case | Method | Status | PSNR | SSIM | LPIPS | Raw rot ° | Aligned rot ° |
|---|---|---|---:|---:|---:|---:|---:|
| SV100-GT | fixed | SUCCESS | 27.117351531982422 | 0.9003052115440369 | 0.07902220636606216 | — | — |
| SV100-GT | naive | SUCCESS | 17.908571243286133 | 0.7507554292678833 | 0.14861881732940674 | 0.17414969937534208 | 0.14286414140689588 |
| SV100-GT | camp | SUCCESS | 19.900388717651367 | 0.7828549742698669 | 0.12358926236629486 | 0.179176459151136 | 0.12297168531717992 |
| SV100-T0.5 | fixed | SUCCESS | 23.16393280029297 | 0.8242353796958923 | 0.19025984406471252 | — | — |
| SV100-T0.5 | naive | SUCCESS | 17.72048568725586 | 0.75315260887146 | 0.14314380288124084 | 0.16016191246733955 | 0.15316289264650607 |
| SV100-T0.5 | camp | SUCCESS | 16.106420516967773 | 0.732803463935852 | 0.18781548738479614 | 0.1242628895735921 | 0.11853224280988138 |
| SV100-T1 | fixed | SUCCESS | 20.352632522583008 | 0.7836681604385376 | 0.24935901165008545 | — | — |
| SV100-T1 | naive | SUCCESS | 15.900127410888672 | 0.7369078993797302 | 0.1968562752008438 | 0.37691880466686223 | 0.25735811810525144 |
| SV100-T1 | camp | SUCCESS | 16.826087951660156 | 0.7462030053138733 | 0.17205750942230225 | 0.49968487667531386 | 0.1411148348250654 |
| SV100-T2 | fixed | SUCCESS | 17.410005569458008 | 0.7545003890991211 | 0.3104111850261688 | — | — |
| SV100-T2 | naive | SUCCESS | 16.844139099121094 | 0.747525691986084 | 0.1714111864566803 | 0.3576907654318797 | 0.17645954241835496 |
| SV100-T2 | camp | SUCCESS | 19.009138107299805 | 0.7670914530754089 | 0.13473717868328094 | 0.17372837416469378 | 0.14094812948916072 |
| SV100-T5 | fixed | SUCCESS | 14.144984245300293 | 0.7271924018859863 | 0.38807252049446106 | — | — |
| SV100-T5 | naive | SUCCESS | 17.533849716186523 | 0.7530051469802856 | 0.15946823358535767 | 0.30762172229369367 | 0.3024573854516597 |
| SV100-T5 | camp | SUCCESS | 19.71722412109375 | 0.7721560597419739 | 0.1357286274433136 | 0.15153271372571678 | 0.12746115740062552 |
| SV100-R0.5 | fixed | SUCCESS | 20.094165802001953 | 0.7804862260818481 | 0.24594688415527344 | — | — |
| SV100-R0.5 | naive | SUCCESS | 20.562286376953125 | 0.793068528175354 | 0.11532619595527649 | 0.14250513597009293 | 0.10354943495223193 |
| SV100-R0.5 | camp | SUCCESS | 18.255605697631836 | 0.7683975696563721 | 0.1436251699924469 | 0.2220425716671775 | 0.14080875617020408 |
| SV100-R1 | fixed | SUCCESS | 17.114395141601562 | 0.7534263134002686 | 0.31077659130096436 | — | — |
| SV100-R1 | naive | SUCCESS | 15.528212547302246 | 0.7332556247711182 | 0.20048755407333374 | 0.2700870145416624 | 0.2634725693301638 |
| SV100-R1 | camp | SUCCESS | 16.700990676879883 | 0.7418728470802307 | 0.16722877323627472 | 0.1793806701179328 | 0.12620268885861477 |
| SV100-R2 | fixed | SUCCESS | 14.811619758605957 | 0.7333740592002869 | 0.3723008334636688 | — | — |
| SV100-R2 | naive | SUCCESS | 16.13628387451172 | 0.7383190989494324 | 0.1885874718427658 | 0.30028402066716964 | 0.21096732084116376 |
| SV100-R2 | camp | SUCCESS | 19.151582717895508 | 0.763823390007019 | 0.13089749217033386 | 0.2058375743406228 | 0.1255037679640463 |
| SV100-R5 | fixed | SUCCESS | 12.007059097290039 | 0.6993001103401184 | 0.47173792123794556 | — | — |
| SV100-R5 | naive | SUCCESS | 17.4700870513916 | 0.7498352527618408 | 0.16173501312732697 | 0.41562610770552155 | 0.22216747809338858 |
| SV100-R5 | camp | SUCCESS | 15.084413528442383 | 0.7246410846710205 | 0.22007404267787933 | 0.38123366847465284 | 0.13798784586962257 |

## SV100-GT

[Convergence curves](figures/SV100-GT_curves.png)

naive: oscillation candidate; inspect curves and camera updates.
naive: test PSNR is below the fixed-pose baseline.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: test PSNR is below the fixed-pose baseline.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-GT_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-GT_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-GT_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-GT_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-GT_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-GT_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-GT_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-GT_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-GT_view_100_accumulation.png)

## SV100-T0.5

[Convergence curves](figures/SV100-T0.5_curves.png)

naive: oscillation candidate; inspect curves and camera updates.
naive: test PSNR is below the fixed-pose baseline.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: test PSNR is below the fixed-pose baseline.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-T0.5_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-T0.5_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-T0.5_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-T0.5_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-T0.5_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-T0.5_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-T0.5_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-T0.5_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-T0.5_view_100_accumulation.png)

## SV100-T1

[Convergence curves](figures/SV100-T1_curves.png)

naive: late validation stagnation/decline candidate (batch-sampling noise possible).
naive: oscillation candidate; inspect curves and camera updates.
naive: test PSNR is below the fixed-pose baseline.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: test PSNR is below the fixed-pose baseline.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-T1_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-T1_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-T1_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-T1_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-T1_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-T1_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-T1_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-T1_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-T1_view_100_accumulation.png)

## SV100-T2

[Convergence curves](figures/SV100-T2_curves.png)

naive: oscillation candidate; inspect curves and camera updates.
naive: test PSNR is below the fixed-pose baseline.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-T2_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-T2_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-T2_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-T2_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-T2_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-T2_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-T2_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-T2_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-T2_view_100_accumulation.png)

## SV100-T5

[Convergence curves](figures/SV100-T5_curves.png)

naive: oscillation candidate; inspect curves and camera updates.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.

[View 0 rgb comparison](figures/SV100-T5_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-T5_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-T5_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-T5_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-T5_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-T5_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-T5_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-T5_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-T5_view_100_accumulation.png)

## SV100-R0.5

[Convergence curves](figures/SV100-R0.5_curves.png)

naive: late validation stagnation/decline candidate (batch-sampling noise possible).
naive: oscillation candidate; inspect curves and camera updates.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: test PSNR is below the fixed-pose baseline.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-R0.5_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-R0.5_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-R0.5_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-R0.5_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-R0.5_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-R0.5_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-R0.5_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-R0.5_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-R0.5_view_100_accumulation.png)

## SV100-R1

[Convergence curves](figures/SV100-R1_curves.png)

naive: oscillation candidate; inspect curves and camera updates.
naive: test PSNR is below the fixed-pose baseline.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: test PSNR is below the fixed-pose baseline.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-R1_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-R1_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-R1_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-R1_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-R1_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-R1_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-R1_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-R1_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-R1_view_100_accumulation.png)

## SV100-R2

[Convergence curves](figures/SV100-R2_curves.png)

naive: oscillation candidate; inspect curves and camera updates.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-R2_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-R2_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-R2_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-R2_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-R2_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-R2_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-R2_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-R2_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-R2_view_100_accumulation.png)

## SV100-R5

[Convergence curves](figures/SV100-R5_curves.png)

naive: oscillation candidate; inspect curves and camera updates.
naive: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.
camp: oscillation candidate; inspect curves and camera updates.
camp: image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.

[View 0 rgb comparison](figures/SV100-R5_view_000_rgb.png)

[View 0 depth comparison](figures/SV100-R5_view_000_depth.png)

[View 0 accumulation comparison](figures/SV100-R5_view_000_accumulation.png)

[View 50 rgb comparison](figures/SV100-R5_view_050_rgb.png)

[View 50 depth comparison](figures/SV100-R5_view_050_depth.png)

[View 50 accumulation comparison](figures/SV100-R5_view_050_accumulation.png)

[View 100 rgb comparison](figures/SV100-R5_view_100_rgb.png)

[View 100 depth comparison](figures/SV100-R5_view_100_depth.png)

[View 100 accumulation comparison](figures/SV100-R5_view_100_accumulation.png)
