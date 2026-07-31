# OpVQ balance vs balanceNo 分析

分析日期：2026-05-17

## 对比对象

- `2026.5.16_opvq_2code_fromMulBasedSps_lr3e4_balance`
- `2026.5.16_opvq_2code_fromMulBasedSps_lr3e4_balanceNo`

两个实验的结构、学习率、数据 split、symm loss 都相同；核心区别只有：

| 实验 | balance loss | symm loss |
|---|---:|---:|
| `balance` | on, scalar = 0.01 | on, scalar = 0.05 |
| `balanceNo` | off | on, scalar = 0.05 |

所以这组实验主要回答：OpVQ 里没有 balance loss 时，2-code VQ 是否会 collapse。

## Best checkpoint 对齐

规则：只在 eval 可见 epoch 中选 checkpoint，并取对应 `Train_record.txt` 里 `total_loss` 最低的 epoch。

| 实验 | best epoch | train add | train mm21 | train H | eval add | eval mm21 | eval H | eval hard code usage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `balance` | 25000 | 0.739 | 0.838 | 0.785 | 0.286 | 0.199 | 0.234 | code1 0.218 / code2 0.782 |
| `balanceNo` | 52500 | 0.543 | 0.744 | 0.628 | 0.016 | 0.103 | 0.028 | code1 1.000 / code2 0.000 |

结论很直接：关掉 balance 后，训练集还能靠单一 code 拟合到中等水平，但 eval 基本崩掉；balance 版本虽然没有学出干净的 add/mm21 code 分化，但至少避免了完全 one-code collapse，并且 train/eval 都显著更好。

## Eval 曲线现象

`balance` 的稳定区间里，eval harmonic 最好是 epoch 37500：

| epoch | eval add | eval mm21 | eval H | hard code usage |
|---:|---:|---:|---:|---|
| 25000 | 0.286 | 0.199 | 0.234 | code1 0.218 / code2 0.782 |
| 37500 | 0.222 | 0.338 | 0.268 | code1 0.213 / code2 0.787 |
| 57500 | 0.190 | 0.368 | 0.251 | code1 0.208 / code2 0.792 |

epoch 60000 的 eval harmonic 形式上到 0.306，但不能当成有效最好结果：train 从 epoch 58000 开始出现数值爆炸，`op_vq_loss`、`sps_vq_loss`、`symm_loss` 都跃迁到巨大值，eval@60000 的 `total_loss` 也到 `1.18e13`。这个点更像坏状态下的偶然 nearest-label accuracy，而不是稳定泛化。

`balanceNo` 从头到尾 hard assignment 都是 code1=1.0/code2=0.0，eval harmonic 最高也只有 0.082（epoch 2500）。这说明没有 balance 时，OpEncoder + VQ codebook 没有自然动力去使用第二个 code。

## Code 分化质量

`balance` 不是完美解决。它的 `code_entropy` 在中后期接近 0.693，看起来像 soft probability 很平衡，但 hard assignment 仍长期约为 code1 0.21 / code2 0.79。这说明当前 balance loss 主要约束 `softmax(-distance/tau)` 的 batch mean，而不保证 argmin 后的 hard code usage 真正平衡。

在 best epoch 25000 上：

| split | add code1 | add code2 | mm21 code1 | mm21 code2 |
|---|---:|---:|---:|---:|
| train | 0.049 | 0.951 | 0.253 | 0.747 |
| eval | 0.143 | 0.857 | 0.250 | 0.750 |

这有一点点 operation preference：code1 更常出现在 mm21 上。但 code2 仍然吃掉 add 和 mm21 的大多数样本，所以还不是“q1=add, q2=mm21”这种清晰语义分化。

## 主要判断

1. balance loss 是必要的。没有它时 hard code 直接塌缩成单 code，eval 泛化几乎不可用。
2. 当前 soft balance 不够。它能让 code probability 看起来平衡，却不能保证 hard assignment 平衡，也不能保证语义分化。
3. symm loss 没有破坏 `balanceNo` 的稳定性，但在 `balance` run 后期出现了严重数值爆炸。这个更像组合式 symm regularization 在长训练后把 decoder/SPS-VQ 路径推到不稳定区域。
4. `balance` 比 `balanceNo` 明显更值得继续，但要配合 early stopping 或更稳的 symm 配置。

## 下一步建议

- 先把 `balance` 的有效比较窗口限制在 epoch 57500 之前，训练时保存并优先分析 early-stopped checkpoint。
- 新增一个 `symm_loss_scalar=0.01` 的版本，验证后期爆炸是否来自 symm 太强。
- 做一个 `include_sps_vq_loss=False` 的 symm ablation，确认爆炸是不是由 repeated decoder outputs 经过 frozen SPS VQ 的 loss 放大引起。
- balance 目标可以改成 hard/near-hard assignment balance，例如对 argmin usage 的 moving average 做约束，或降低 `softmax_tau`，避免 soft probability 平衡但 hard code collapse/偏置。
- 如果目标是 add/mm21 语义分化，后续还需要专门看 critical pair 或 pairwise op assignment；当前整体 accuracy 和 code rate 只能说明“有没有 collapse”，还不能证明语义已经干净分开。
