# FiLM-conditioned OperNet

## 直观解释

旧的 concat-query 模式把 query 只拼到输入：

```text
[e_a, e_b, q] -> MLP -> z_pred -> VQ -> e_pred
```

这样 query 只在第一层进入网络。后续层的权重完全共享，所以网络容易学成一个主要计算路径，query 只提供轻微扰动。

FiLM 模式把 query 用到每个 hidden layer。每一层都有一个小的 query-to-FiLM 线性层，生成 `gamma` 和 `beta`，用来调制该层 hidden activation：

```text
hidden -> (1 + gamma(q)) * hidden + beta(q)
```

所以 q0/q1 不只是输入的一部分，而是在每一层改变中间表征的尺度和偏置，更像是在同一个 shared `OperNet` 内选择不同计算路径。

## 符号

令：

- `d` 是 content embedding 维度。
- `m` 是 query 维度。
- `e_a, e_b, e_c in R^d`。
- `q in R^m`。
- `x = [e_a; e_b] in R^(2d)`。
- hidden width 为 `H`。

## FiLM 层

第 `l` 层由 query 生成调制参数：

```text
[gamma_l(q); beta_l(q)] = A_l q + c_l
```

其中：

```text
gamma_l(q), beta_l(q) in R^H
```

FiLM 调制为：

```text
FiLM_l(u, q) = (1 + gamma_l(q)) o u + beta_l(q)
```

`o` 表示逐元素乘法。

## 前向传播

第一层：

```text
u_0 = W_0 x + b_0
h_0 = ReLU(FiLM_0(u_0, q))
```

后续 hidden 层：

```text
u_l = W_l h_(l-1) + b_l
h_l = ReLU(FiLM_l(u_l, q))
```

输出到连续 operator latent：

```text
z_pred = W_out h_(L-1) + b_out
```

再使用冻结 SPS/VQ codebook quantize：

```text
e_pred = VQ(z_pred)
```

## sanity check loss

当前 sanity check 目标是先证明一个 shared `OperNet` 能同时做 add 和 mul_mod21。

对同一个 `(e_a, e_b)` 分别用 q0/q1：

```text
e_0 = OperNet(e_a, e_b, q0)
e_1 = OperNet(e_a, e_b, q1)
```

逐样本误差：

```text
L_0 = MSE(e_0, e_c)
L_1 = MSE(e_1, e_c)
```

sanity check 中用标签指定 query：

```text
L_oper = mean( L_0 if c = a + b else L_1 )
```

训练总 loss 还包含 VQ regularization 和可选的 symmetry/associativity regularization：

```text
L_total = L_oper + lambda_vq L_vq + L_symm
```

## 初始化

`film_init_identity=True` 时，所有 FiLM 层初始输出 `gamma=0, beta=0`，所以初始行为接近普通 shared MLP：

```text
FiLM_l(u, q) = u
```

训练开始后，FiLM 层会从梯度中学习 q0/q1 对每层 activation 的不同调制。

这个初始化让新模型一开始不至于过强扰动，同时保留后续分化成 add/mul 两条路径的能力。

