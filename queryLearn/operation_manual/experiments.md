# queryLearn 实验记录

## 通用 diagnostics

从 2026-05-14 起，训练会额外写出关键 pair 的 winner-take-all assignment diagnostics。训练 loss 不变；这些字段只用于观察 `sanity_check=False` 时，同一个 `(a,b)` 同时存在 add target 和 mm21 target 时，q1/q2 是否发生单 query 独占。

输出文件：

```text
CriticalPairStats_record.csv
```

含义：

- 启动训练后先扫描 train loader，找出关键 pair：同一个 `(a,b)` 在训练集中同时出现 add 样本 `(a,b,a+b)` 和 mm21 样本 `(a,b,(a*b)%21)`。
- 只统计训练集，不统计 eval。
- 每个 epoch 中，对每个关键 pair 分别看 add 样本和 mm21 样本由 q1 还是 q2 的 loss 更小。
- 如果 add 和 mm21 都由 q1 赢，记一次 `q1_exclusive`；如果都由 q2 赢，记一次 `q2_exclusive`。
- 如果 add/mm21 分别由不同 query 赢，记一次 `split`，这是符合预期的分化信号。
- 如果缺少某个 operation、两个 query 打平、或 operation 内 winner 无法唯一判定，记一次 `tie_or_missing`。
- 每个 `log_interval` 写一次 CSV，每行是一个关键 pair 在该 interval 内跨多个 epoch 的独占率和计数。

## 2026-05-10 - FiLM sanity check from mul-based SPS

实验名：

```text
2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_fc_5_1024
```

配置路径：

```text
S3Plus/queryLearn/exps/2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_fc_5_1024/config.py
```

目的：

- 先不解决 query 自发分化问题。
- 在 `sanity_check=True` 下强制 q1 负责 add，q2 负责 non-add/mm21。
- 验证 FiLM-conditioned shared `OperNet` 是否能同时表示 add 和 mm21 两条计算路径。

关键配置：

```python
'query_learner': {
    'query_dim': 2,
    'train_queries': False,
}
'operator': {
    'condition_mode': 'film',
    'film_init_identity': True,
    'unit': 1024,
    'n_hidden_layers': 5,
}
'sanity_check': True
```

运行命令：

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
cd /home/xuanjie.liu/Projects/S3Plus/queryLearn
python batch_train.py 2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_fc_5_1024
exit
```

重点看：

- `Eval_record.txt` 中 `add_acc_q1` 是否上升。
- `Eval_record.txt` 中 `mm21_acc_q2` 是否上升。
- `add_acc` 和 `mm21_acc` 是否能同时达到比 concat-query baseline 更好的水平。
- `EvalResults/query_operation_*` 表格里，q1/q2 是否形成稳定的 add / `*m` 区域。

相关分析报告：

```text
S3Plus/queryLearn/experiment_analysis/compare_sanity_check_2026-05-10.html
```

该报告比较了本实验、mul-based concat baseline 和 add-based concat baseline，使用三组实验的 `1/Eval_record.txt`。

已知注意点：

- `query_learner.query_dim` 现在已被 `QueryLearn.py` 正确读取；旧字段 `in_dim` 仍作为 fallback。
- `train_sps` 配置目前不控制 SPS 是否训练；`QueryLearn` 会冻结加载的 SPS/VQ 模型。
- 当前 sanity check 分配逻辑是 `a + b == c` 走 q1，否则走 q2。mm21 数据集中若某些样本也满足 add，会被当作 add/ambiguous 处理。

结果简析：

- 比 concat-query baseline 明显更好，说明 FiLM 确实让 query 更有效地改变了 shared `OperNet` 的计算路径。
- 训练集很快达到高准确率，约 10k-20k 时 train add/mul 已经接近或超过 0.94。
- eval 在 10k 左右 mul 最好，在 15k 左右 add 最好，但后期退化明显。
- 这更像是高学习率、较强 symmetry regularization 和后期过拟合/漂移共同造成的问题，而不是 FiLM 容量不足。

## 2026-05-10 - FiLM sanity check, lower LR, weaker symm, fixed queries

实验名：

```text
2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_lr3e4_symm005_fixedQ
```

配置路径：

```text
S3Plus/queryLearn/exps/2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_lr3e4_symm005_fixedQ/config.py
```

目的：

- 继续 sanity check。
- 在保留 FiLM capacity 的前提下，让优化更稳、减少后期 eval 退化。
- 尽量判断上一轮“不够完美”主要来自优化/正则，而不是模型表达能力。

相对上一轮的改动：

```python
'query_learner': {
    'query_dim': 2,
    'init_queries': [[1.0, -1.0], [-1.0, 1.0]],
    'train_queries': False,
}
'learning_rate': 3e-4
'optimizer': 'adamw'
'weight_decay': 1e-4
'grad_clip_norm': 1.0
'checkpoint_interval': 2500
'checkpoint_after': 5000
'eval_interval': 2500
'symm_loss_scalar': 0.05
```

运行命令：

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
cd /home/xuanjie.liu/Projects/S3Plus/queryLearn
python batch_train.py 2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_lr3e4_symm005_fixedQ
exit
```

重点看：

- 10k-30k 区间 eval 是否比上一轮更平滑。
- `add_acc_q1` 和 `mm21_acc_q2` 是否能同时维持高值，而不是一个上升时另一个掉。
- `oper_loss` 后期是否仍明显上升。
- 如果最佳点仍集中在前 15k，下一步可以考虑早停或显式选择 best checkpoint。
