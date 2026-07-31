# queryLearn 模型结构

这个目录用自然语言、公式和图记录 `queryLearn` 的模型结构。

文件：

- `film_opernet.md`：FiLM 模式说明、公式、训练目标。
- `film_opernet.svg`：模型结构图，可直接在浏览器或 Markdown preview 中查看。
- `concat_opernet.svg`：concat query conditioning baseline 的模型结构图。

当前问题设定：

- 输入样本是三张图片 `(a, b, c)`。
- 冻结的 VQ/SPS encoder 将图片编码为 content embedding：`e_a, e_b, e_c`。
- `OperNet` 接收 `(e_a, e_b, q)`，输出预测的 content embedding `e_pred`。
- q0/q1 表示不同 query，希望最终能对应不同运算。

无监督训练支持两种 operation objective。默认 hard-min 为
`mean(min(mse_q1, mse_q2))`。可选的等权 Gaussian mixture NLL 使用

`-tau * log(0.5 * exp(-mse_q1 / tau) + 0.5 * exp(-mse_q2 / tau))`,

其中 `tau = 2 * variance / content_dim`。这个形式与固定各向同性方差的
Gaussian mixture NLL 有相同的 responsibilities 和最优解，同时保持与原 MSE
接近的梯度尺度。
