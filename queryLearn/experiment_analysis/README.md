# queryLearn 实验分析通用规则

这个目录存放指定实验的静态分析报告。报告优先使用已有 `Train_record.txt`、`Eval_record.txt` 和 `TrainingResults/`、`EvalResults/` 中的 query 可视化图片，不启动训练。

每次分析应生成一个独立子目录：

```text
experiment_analysis/<report_name>/index.html
experiment_analysis/<report_name>/assets/<exp_short>/*.png
```

HTML 只引用这个子目录内的相对路径图片。这样整个 `<report_name>/` 文件夹下载到本地后，可以直接打开 `index.html` 查看。

## 生成脚本

默认生成当前 sanity-check 对比报告：

```bash
cd /home/xuanjie.liu/Projects/S3Plus/queryLearn
python experiment_analysis/generate_analysis.py
```

自定义报告可重复传入 `--experiment`：

```bash
python experiment_analysis/generate_analysis.py \
  --report-name my_report \
  --title "My queryLearn report" \
  --experiment "short_name|experiment_dir_name|Display label|#0f766e"
```

`experiment_dir_name` 是 `S3Plus/queryLearn/exps/` 下的目录名。当前脚本默认使用 sub-exp `1`。

## Best Epoch 选择

分析报告中的 best epoch 统一按训练记录选择：

1. 先取 `Eval_record.txt` 中存在的 epoch 集合。
2. 在这些 epoch 中查找对应 `Train_record.txt` 的 `total_loss`。
3. 选择 `total_loss` 最低的 epoch 作为该实验的 best epoch。
4. 所有柱状图、summary 表格和 query pairwise 图片都使用同一个 best epoch。

如果某次分析明确要用其他 loss，例如 `oper_loss`，必须在对应 HTML 报告中单独说明。

## 柱状图规则

比较多个实验时：

- 颜色只表示实验。
- x 轴固定为三组：`add_acc`、`mul_acc`、`harmonic mean`。
- 每组中每个实验画一根柱子。
- `harmonic mean = 2 * add_acc * mul_acc / (add_acc + mul_acc)`。
- train / eval 结果用页签切换，但使用同一个 best epoch。

## Query Pairwise 可视化规则

query pairwise 可视化使用每个实验 best epoch 对应的图片：

- train 页签读取 `TrainingResults/query_operation_train_epoch_<epoch>_q*.png`。
- eval 页签读取 `EvalResults/query_operation_eval_epoch_<epoch>_q*.png`。
- 同一实验展示 q1 和 q2 两张图。
- 生成脚本会把这些图片复制到报告子目录的 `assets/` 中，HTML 不直接引用原始实验目录。
- Query Pairwise Visualization section 按实验纵向排列，不把多个实验横向并排。
- 每一行只展示一个实验；该行内 q1 和 q2 两张图并排展示。
- 在窄屏幕上，q1 和 q2 可以自动改为上下排列，避免图片或文字溢出。
- 每张 query 图片都应支持点击放大。放大视图使用当前报告目录内的相对路径图片，并提供关闭方式，例如点击背景、关闭按钮或 Esc。

## 页签规则

所有结果 section 都应支持 train / eval 切换。页签只切换展示的数据来源，不改变 best epoch 选择规则。
