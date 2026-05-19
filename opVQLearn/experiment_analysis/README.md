# opVQLearn 实验分析规则

本目录沿用 `queryLearn/experiment_analysis/README.md` 的报告组织方式：

```text
experiment_analysis/<report_name>/index.html
experiment_analysis/<report_name>/assets/<exp_short>/*.png
```

报告应优先使用已有 `Train_record.txt`、`Eval_record.txt`、`TrainingResults/` 和 `EvalResults/`，不启动训练。HTML 只引用报告子目录内的相对路径图片，方便下载整个报告目录后本地打开。

## OpVQ 特殊规则

- best epoch 仍按 queryLearn 规则选择：先取 eval 可见 epoch，再选对应 train `total_loss` 最低的 epoch。
- 整体柱状图仍展示 `add_acc`、`mm21_acc` 和 harmonic mean。
- OpVQ record 没有 query-wise accuracy 字段，因此 query-wise section 改为 code usage 诊断：
  - `code1_rate` / `code2_rate`
  - `add_code1_rate` / `add_code2_rate`
  - `mm21_code1_rate` / `mm21_code2_rate`
- data-pair 可视化使用 OpVQ 图片命名：
  - train: `TrainingResults/op_assignment_train_epoch_<epoch>_add.png`
  - train: `TrainingResults/op_assignment_train_epoch_<epoch>_mm21.png`
  - eval: `EvalResults/op_assignment_eval_epoch_<epoch>_add.png`
  - eval: `EvalResults/op_assignment_eval_epoch_<epoch>_mm21.png`
- 每个实验纵向展示一行；行内 add set 和 mm21 set 两张图并排，图片支持点击放大。
- 全局 train/eval 切换按钮固定在浏览器窗口右上角。
