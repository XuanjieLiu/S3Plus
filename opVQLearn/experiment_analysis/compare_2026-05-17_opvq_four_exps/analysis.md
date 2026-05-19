# opVQLearn 四实验对比分析

分析日期：2026-05-17

## 核心结论

这四次实验说明了三个点。

1. `balance loss` 是必要的。`no balance + symm0.05 + SPS-in-symm` 全程 hard assignment collapse 到单个 code，best checkpoint 的 eval harmonic 只有 `0.028`。
2. 关闭 symm 后训练拟合最好，但泛化没有同步改善。`balance + no symm` 的 train harmonic 到 `0.879`，但 best checkpoint 的 eval harmonic 只有 `0.028`。
3. 降低 symm 且移除 SPS-in-symm 确实避免了原始实验的天文级 `symm_loss/sps_vq_loss` 爆炸，但 `op_vq_loss` 后期变大，eval harmonic 仍只有 `0.122`。

按 README 的 best epoch 规则，四者中 best-checkpoint eval harmonic 最高的是 `balance + symm0.05 + SPS-in-symm`，为 `0.234`。如果只看未爆炸稳定区间的 eval harmonic 峰值，最高是 `balance + symm0.05 + SPS-in-symm`，epoch `37500` 达到 `0.268`。

## 参数差异

- 原始 `balance + symm0.05 + SPS-in-symm`：有 balance，有强 symm，并把 repeated SPS VQ loss 加进 symm 路径。
- `balanceNo`：去掉 balance，其他和原始相同。
- `symm0.01 no SPS-in-symm`：保留 balance，把 symm scalar 从 `0.05` 降到 `0.01`，且 symm 路径不再累计 SPS VQ loss。
- `no symm`：保留 balance，完全关闭 symm loss。

## 解释

`balanceNo` 的失败非常清楚：没有 balance 时，VQ codebook 没有足够压力使用两个 code，模型退化成单 code decoder。`noSymm` 的训练集结果最好，说明 symm regularization 在当前实现下不是拟合训练集所必需的；但 eval 仍然很低，说明训练拟合和 add/mm21 语义泛化之间仍然有大断层。

`balance + symm0.05 + SPS-in-symm` 在 epoch `58000` 开始数值爆炸，因此 60000 附近的 accuracy 不能作为可信泛化。`balance + symm0.01, no SPS-in-symm` 没有这种天文级爆炸，支持“repeated SPS-in-symm 会放大不稳定性”的判断；但它的 `op_vq_loss` 后期升高，说明移除 SPS-in-symm 后训练仍有另一个 codebook/encoder 对齐问题。

## 下一步

- 保留 balance；没有 balance 的路线可以暂时搁置。
- 重点比较 `noSymm` 和更弱的 symm：当前 symm 没带来泛化收益，下一步应单独验证 `symm_loss_scalar=0.001` 或只在更晚 epoch 打开 symm。
- 给 OpVQ 加 hard-assignment balance 或 moving-average usage balance，因为当前 soft balance 仍允许 hard usage 偏置。
- 新增 pair-level assignment 诊断：整体 accuracy 不能说明 q1/q2 是否真的按 add/mm21 分化。
