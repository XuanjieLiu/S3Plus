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
- 模型/训练逻辑：`S3Plus/queryLearn/QueryLearn.py`
- 实验配置：`S3Plus/queryLearn/exps/<EXP_NAME>/config.py`
- query 可视化：`S3Plus/queryLearn/query_vis.py`
- 模型结构说明：`S3Plus/queryLearn/model_structure/`

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
