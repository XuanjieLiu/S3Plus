# OpVQ 模型结构

## 目标

`OpVQ` 的目标是在不显式监督 add/mm21 identity 的情况下，从 triplet `(a, b, c)` 中学习一个离散 operation code，并让 decoder 使用这个 code 从 `(a, b)` 预测 `c`。

## 表示

冻结的 VQ/SPS 模型先把三张图片编码为 content embedding：

```text
ea, eb, ec in R^d
```

`OpEncoder` 看到完整 triplet：

```text
z_op = OpEncoder([ea, eb, ec])
```

`OpVQCodebook` 有两个可训练 code：

```text
Q = {q1, q2}
q = nearest_Q(z_op)
```

训练时使用 straight-through VQ：

```text
q_st = z_op + stop_grad(q - z_op)
```

`OperDecoder` 只看到输入 pair 和离散 operation code：

```text
z_pred = OperDecoder([ea, eb, q_st])
e_pred = SPS_VQ(z_pred)
```

## Loss

主重建 loss：

```text
L_pred = MSE(e_pred, ec)
```

OpVQ loss：

```text
L_op_vq = ||sg[z_op] - q||^2 + beta ||z_op - sg[q]||^2
```

冻结 SPS 的 VQ commitment loss：

```text
L_sps_vq = SPS_VQ_loss(z_pred)
```

可选 code balance：

```text
p_bar = mean_batch softmax(-dist(z_op, Q) / tau)
L_balance = ||p_bar - [0.5, 0.5]||^2
```

可选 codebook-level symmetry loss：

```text
For each q in {q1, q2}:
  f_q(f_q(a,b), c) ~= f_q(f_q(a,c), b)
  f_q(a, f_q(b,c)) ~= f_q(b, f_q(a,c))
```

这里的 `q1/q2` 是 VQ codebook 中的两个 code embedding，不是由某个样本临时 assignment 出来的 `q`。这个 loss 不使用 add/mm21 label，只要求每个 latent operation code 对应的 decoder dynamics 更像一个交换/对称二元运算。

总 loss：

```text
L_total = L_pred
        + lambda_op * L_op_vq
        + lambda_sps * L_sps_vq
        + lambda_balance * L_balance
        + lambda_symm * L_symm
```

`L_balance` 和 `L_symm` 都由 config 开关控制；关闭时相关项记录为 0，不加入训练 loss。

## 和 queryLearn 的区别

- 不存在 `min(q1_loss, q2_loss)`。
- 不存在手写 q1/q2 assignment。
- q1/q2 是 OpEncoder 通过 VQ codebook 为每个 `(ea, eb, ec)` 推断出的 latent operation code。
- add/mm21 只用于 evaluation、record 和 visualization，不作为训练监督。
- `symm_loss` 只约束 codebook code 的 operation dynamics，也不提供 add/mm21 identity supervision。

## 简图

```text
          ea ----\
                  \
          eb ------> OpEncoder ----> z_op ----> VQ(2 codes) ----> q
                  /                                           |
          ec ----/                                            |
                                                               v
          ea ------------------------------\              [ea, eb, q]
                                            \                 |
          eb --------------------------------> OperDecoder ---+--> pred_ec
```
