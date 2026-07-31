# queryLearn 操作手册

这个目录记录 `queryLearn` 实验怎么跑、每次实验 setup 的关键配置、资源约束和交接信息。以后改 `queryLearn` 训练逻辑、配置或新增实验时，请同步更新这里。

## 资源规则

不要在 login node 上直接跑训练或重计算。先申请 GPU 资源：

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
```

跑完后释放资源：

```bash
exit
```

## 项目目标

当前 `queryLearn` 目标是验证：在冻结的 VQ/SPS 数字表示空间中，单个 shared `OperNet` 是否能仅通过不同 query condition 实现不同二元运算。

当前阶段优先级：

1. 在 `sanity_check=True` 下，先验证 q1/q2 被标签指定时，同一个 `OperNet` 能同时学会 add 和 mm21。
2. 如果 sanity check 成功，再研究无监督情形下 q1/q2 是否能自己分化出 add 和 mm21 的语义。

## 关键入口

- 训练入口：`S3Plus/queryLearn/batch_train.py`
- 训练主流程：`S3Plus/queryLearn/QueryLearn.py`
- OperNet 结构：`S3Plus/queryLearn/opernet.py`
- 训练辅助函数：`S3Plus/queryLearn/training_helpers.py`
- Pair taxonomy / risk 诊断：`S3Plus/queryLearn/pair_diagnostics.py`
- 实验配置：`S3Plus/queryLearn/exps/<EXP_NAME>/config.py`
- query 可视化：`S3Plus/queryLearn/query_vis.py`
- 模型结构说明：`S3Plus/queryLearn/model_structure/`

## 数据 split 和 pair 诊断

当前常用实验会把 `train_data_path` 中的 add train/test 和 mm21 train/test 先合成一个 `ConcatDataset`，再对这个混合后的 dataset 做 `random_split`。因为 `augment_times` 会扩展 dataset length，split 单位是增强后的虚拟样本 index，不是唯一 `(a,b,op)` pair。

训练启动时 `QueryLearn` 会基于实际 train split 建立 pair taxonomy：

- `single_add`：训练集中这个 `(a,b)` 只出现 add。
- `single_mm21`：训练集中这个 `(a,b)` 只出现 mm21。
- `dual_distinct`：训练集中这个 `(a,b)` 同时出现 add 和 mm21，且两个 target 不同。
- `special_ambiguous`：`a + b == (a * b) % 21`，add/mm21 语义不可区分，不进入风险排序。

诊断文件：

- `CriticalPairStats_record.csv`：旧语义，只覆盖 `dual_distinct`，用于分析同一个 query 同时赢走 add/mm21 的情况。
- `PairRiskStats_record.csv`：新语义，同时覆盖 `single_add` / `single_mm21` 的 query 竞争风险，以及 `dual_distinct` 的同 query 独占风险。

## Evaluation 开关

- `use_eval_set` 默认为 `True`。
- 如果配置里设为 `False`，或 random split 后 eval split 为空，`init_dataloaders()` 会返回 `eval_loader=None`。
- `QueryLearn.train()` 看到 `eval_loader=None` 会跳过 pair eval，不写新的 `Eval_record.txt` 行。
- `single_img_eval_set_path` 仍可保留；它用于 query target accuracy 的 embedding lookup，不等价于 pair eval set。

## Record 可视化和 checkpoint

训练会在写入 `Train_record.txt` / `Eval_record.txt` 后同步刷新 record 曲线。可在 config 中配置：

```python
'record_visualizer': {
    'enabled': True,
    'output_dir': 'RecordPlots/',
    'format': 'png',
    'metrics': {
        'query_accuracy': ['add_acc_q1', 'add_acc_q2', 'mm21_acc_q1', 'mm21_acc_q2', 'add_acc', 'mm21_acc'],
        'loss': ['oper_loss', 'hard_min_loss', 'symm_loss', 'total_loss'],
    },
}
```

输出包括每个 metric group 的图片和 `RecordPlots/index.html`。

无监督 operation loss 默认为原有的逐样本 hard-min。Gaussian mixture NLL 可通过 config 开启：

```python
'operation_loss': {
    'type': 'gaussian_mixture_nll',
    'variance': 0.04,
    'mixture_weights': [0.5, 0.5],
    'weight': 1.0,
}
```

GMM 使用固定方差和 mixture prior。训练记录中的 `oper_loss` 是保持 MSE 单位的
scaled NLL；`hard_min_loss` 是同批数据上的原始 winner MSE，用于和 hard-min
实验比较。`sanity_check=True` 时仍使用显式 query 分配，并忽略该无监督 loss
选择。

checkpoint 现在默认只保留两个文件：

- `curr_model.pt`：最近一次写 record 时的模型。
- `best_model.pt`：训练记录中 `best_checkpoint_metric` 最低的模型，默认 `total_loss`。

## 运行模板

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
cd /home/xuanjie.liu/Projects/S3Plus/queryLearn
python batch_train.py <EXP_NAME>
exit
```

## 更新约定

每次新增或修改实验 setup 后，同步更新：

- `operation_manual/experiments.md`：记录实验名、目的、关键配置、运行命令和预期观察指标。
- `model_structure/`：如果模型结构或 loss 有变化，同步更新自然语言说明、公式和图。
- `experiment_analysis/`：如果新增对比分析或静态报告，同步记录报告路径和使用的数据来源。
