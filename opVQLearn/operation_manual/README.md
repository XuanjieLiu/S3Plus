# opVQLearn 操作手册

`opVQLearn` 是和 `queryLearn` 并列的新实验线。它不使用 `min(q1_loss, q2_loss)` 来选择 query，而是学习一个离散 latent operation code：

```text
OpEncoder: [ea, eb, ec] -> q in {q1, q2}
OperDecoder: [ea, eb, q] -> pred_ec
```

## 资源规则

不要在 login node 上跑训练。先申请 GPU 资源：

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
```

跑完释放资源：

```bash
exit
```

## 关键入口

- 训练入口：`S3Plus/opVQLearn/batch_train.py`
- 模型/训练逻辑：`S3Plus/opVQLearn/OpVQLearn.py`
- assignment 可视化：`S3Plus/opVQLearn/op_vis.py`
- 实验配置：`S3Plus/opVQLearn/exps/<EXP_NAME>/config.py`
- 模型结构说明：`S3Plus/opVQLearn/model_structure/opvq.md`

## 运行模板

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
cd /home/xuanjie.liu/Projects/S3Plus/opVQLearn
python batch_train.py 2026.5.16_opvq_2code_fromMulBasedSps_lr3e4_balance
exit
```

## 当前推荐下一轮实验

```text
2026.5.17_opvq_2code_fromMulBasedSps_lr3e4_balance_symm001_noSpsVqSymm
```

运行：

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
cd /home/xuanjie.liu/Projects/S3Plus/opVQLearn
python batch_train.py 2026.5.17_opvq_2code_fromMulBasedSps_lr3e4_balance_symm001_noSpsVqSymm
exit
```

这个 setup 用来验证 `2026.5.16_opvq_2code_fromMulBasedSps_lr3e4_balance` 后期数值爆炸是否来自过强的 symm regularization。它保留 balance loss，把 `symm.loss_scalar` 从 `0.05` 降到 `0.01`，并关闭 symm 路径里的 SPS VQ loss。

## 首个实验

```text
2026.5.16_opvq_2code_fromMulBasedSps_lr3e4_balance
```

当前配置已启用 balance loss 和 codebook-level symm loss。

重点看：

- `Train_record.txt` / `Eval_record.txt` 中 `add_acc`、`mm21_acc` 是否同时上升。
- `symm_loss` 是否稳定下降；它约束两个 VQ code 的 decoder dynamics 满足交换/对称结构，不使用 add/mm21 标签。
- `code1_rate` / `code2_rate` 是否避免塌缩到单个 code。
- `add_code*_rate` 与 `mm21_code*_rate` 是否形成互补。
- `TrainingResults/` 和 `EvalResults/` 中 `op_assignment_*_add.png`、`op_assignment_*_mm21.png` 是否显示稳定的 operation-code 分化。

## Symm loss 配置

推荐起始配置：

```python
'symm': {
    'use_symm_loss': True,
    'loss_scalar': 0.05,
    'include_sps_vq_loss': True,
}
```

`symm_loss` 直接作用于 codebook 中的 `q1/q2` 两个 code embedding。它不会根据样本的 add/mm21 类型指定 code identity，只会要求每个 code 对应的 operation dynamics 更接近交换/对称二元运算。若发现 reconstruction 明显变慢或 `add_acc/mm21_acc` 被压低，可先把 `loss_scalar` 降到 `0.01`。

## 可视化规则

- 每个 epoch 输出 add set 和 mm21 set 两张图。
- 格子背景：浅绿表示该格所有 observed samples 都预测正确，浅红表示至少一个 observed sample 预测错误，白色表示该 epoch 没有这个 data point。
- 格子文字：`1` 或 `2` 表示 OpEncoder 选择的 VQ code；若同格重复样本出现两个 code，则显示 `1,2`。
- 特殊 pair，即 `a + b == (a * b) % 21`，使用加粗边框，不使用紫色底色。
