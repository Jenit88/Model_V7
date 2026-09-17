# Training report - Model_v7_scratch_rtx

Generated 2026-09-12T12:29:15+00:00

## Where the run is

- Epochs completed: **49** of 120
- Learning rate: 2.091e-04
- Train loss: 0.1595  |  validation loss: 0.1691
- Foreground mIoU: train 0.9007  |  validation 0.9044

## Best checkpoint so far

- Selected on **mask_map50_95** = **0.8407** at epoch 50
- Instance precision 0.9575, recall 0.9797, F1 0.9685 at IoU 0.50
- Mask mAP50 0.9806, mAP50-95 0.8407
- TP 11,121 / FP 494 / FN 230

### Per class, at the best checkpoint

| class | TP | FP | FN | P | R | F1 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|---|
| Rectangle | 2,751 | 230 | 59 | 0.9228 | 0.9790 | 0.9501 | 0.9843 | 0.8489 |
| Rectangle_concave | 79 | 8 | 4 | 0.9080 | 0.9518 | 0.9294 | 0.9738 | 0.8931 |
| circle | 1,560 | 95 | 58 | 0.9426 | 0.9642 | 0.9533 | 0.9788 | 0.8287 |
| circle_full | 6,731 | 161 | 109 | 0.9766 | 0.9841 | 0.9803 | 0.9855 | 0.7920 |

### Evaluation history

| epoch | P | R | F1 | mAP50 | mAP50-95 | saved |
|---|---|---|---|---|---|---|
| 1 | 0.0399 | 0.2485 | 0.0688 | 0.0617 | 0.0229 | yes |
| 5 | 0.7848 | 0.8982 | 0.8377 | 0.8988 | 0.6396 | yes |
| 10 | 0.9172 | 0.9564 | 0.9364 | 0.9590 | 0.7705 | yes |
| 15 | 0.9407 | 0.9697 | 0.9550 | 0.9753 | 0.8072 | yes |
| 20 | 0.9478 | 0.9720 | 0.9597 | 0.9777 | 0.8153 | yes |
| 25 | 0.9470 | 0.9752 | 0.9609 | 0.9782 | 0.8242 | yes |
| 30 | 0.9521 | 0.9736 | 0.9627 | 0.9781 | 0.8294 | yes |
| 35 | 0.9517 | 0.9767 | 0.9640 | 0.9788 | 0.8314 | yes |
| 40 | 0.9539 | 0.9776 | 0.9656 | 0.9793 | 0.8374 | yes |
| 45 | 0.9518 | 0.9784 | 0.9649 | 0.9792 | 0.8362 |  |
| 50 | 0.9575 | 0.9797 | 0.9685 | 0.9806 | 0.8407 | yes |

## What the numbers say

- Still improving: mask_map50_95 is rising about 0.0027 per evaluation over the last 5.

## Semantic IoU by class (last epoch)

| class | train IoU | val IoU |
|---|---|---|
| Rectangle | - | - |
| Rectangle_concave | - | - |
| circle | - | - |
| circle_full | - | - |
