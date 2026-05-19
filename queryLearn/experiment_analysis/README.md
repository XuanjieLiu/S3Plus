# queryLearn 实验分析通用规则

这个目录存放指定实验的静态分析报告。报告优先使用已有 `Train_record.txt`、`Eval_record.txt` 和 `TrainingResults/`、`EvalResults/` 中的 query 可视化图片，不启动训练。

每次分析应生成一个独立子目录：

```text
experiment_analysis/<report_name>/index.html
experiment_analysis/<report_name>/assets/<exp_short>/*.png
```

HTML 只引用这个子目录内的相对路径图片。这样整个 `<report_name>/` 文件夹下载到本地后，可以直接打开 `index.html` 查看。

多 sub-exp 聚合报告中，图片会按重复实验再分一层目录：

```text
experiment_analysis/<report_name>/assets/<exp_short>/sub<id>/*.png
```

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
  --analysis-file experiment_analysis/my_report/analysis.md \
  --experiment "short_name|experiment_dir_name|Display label|#0f766e"
```

如果一个实验目录下有多个 sub-exp，并希望按重复实验聚合：

```bash
python experiment_analysis/generate_analysis.py \
  --report-name my_repeated_report \
  --title "My repeated-run report" \
  --analysis-file experiment_analysis/my_repeated_report/analysis.md \
  --sub-exps 1,2,3 \
  --experiment "short_name|experiment_dir_name|Display label|#0f766e"
```

`experiment_dir_name` 是 `S3Plus/queryLearn/exps/` 下的目录名。当前脚本默认使用 sub-exp `1`。

## 报告顶部分析规则

HTML 顶部首先应展示实验分析正文，而不是画图规则或操作说明。读者打开报告时，应先看到这次实验说明了什么。

分析正文建议包含：

- 核心结论：哪个实验更好，关键指标差多少。
- 参数差异：本次对比调了哪些关键参数。
- 可能原因：这些参数为什么可能有用，以及哪些判断只是推测。
- 风险和限制：例如多个因素同时变化时，不能断言单个因素的因果贡献。
- 下一步验证：下一组 ablation 或更接近目标设定的实验。

具体报告的分析正文建议写在该报告子目录的 `analysis.md` 中，并通过 `--analysis-file` 传入生成脚本。README 只记录通用规则，不记录某次具体实验结论。

## Best Epoch 选择

分析报告中的 best epoch 统一按训练记录选择：

1. 先取 `Eval_record.txt` 中存在的 epoch 集合。
2. 在这些 epoch 中查找对应 `Train_record.txt` 的 `total_loss`。
3. 选择 `total_loss` 最低的 epoch 作为该实验的 best epoch。
4. 所有柱状图、summary 表格和 query pairwise 图片都使用同一个 best epoch。

如果某次分析明确要用其他 loss，例如 `oper_loss`，必须在对应 HTML 报告中单独说明。

多 sub-exp 报告中，每个 sub-exp 独立按上述规则选择 best epoch；聚合图表对这些 best checkpoint 的指标计算 mean/std。

## 柱状图规则

比较多个实验时：

- 颜色只表示实验。
- x 轴固定为三组：`add_acc`、`mm21_acc`、`harmonic mean`。
- 每组中每个实验画一根柱子。
- `harmonic mean = 2 * add_acc * mm21_acc / (add_acc + mm21_acc)`。
- train / eval 结果用页签切换，但使用同一个 best epoch。

## 折线图规则

- 折线图横轴必须根据当前报告中实际最大 epoch 自动缩放。
- 不要固定用 50000 作为横轴上限，否则更长训练的曲线会跑出绘图区并进入 legend 区域。
- 线条右侧的文字标注应放在绘图区外的保留区域中，曲线本身必须限制在绘图区内。

## Query-wise 柱状图规则

每份分析报告都应包含 query-wise best epoch accuracy 对比：

- 这个 section 单独展示，不和整体 accuracy 柱状图混在一起。
- train / eval 结果用页签切换，但使用同一个 best epoch。
- 每行只展示一个实验。
- 每个实验画两组柱子：`add_acc` 和 `mm21_acc`。
- 每组包含三根柱子：`q1_acc`、`q2_acc` 和 `overall`。
- record 字段中 `add_acc_q1` / `mm21_acc_q1` 对应图像和报告中的 `q1_acc`。
- record 字段中 `add_acc_q2` / `mm21_acc_q2` 对应图像和报告中的 `q2_acc`。
- `overall` 使用 record 中已有的 `add_acc` 或 `mm21_acc`。
- 分析旧实验时，生成脚本可以兼容读取历史字段：`add_acc_q0` 会映射到 `add_acc_q1`，`mul_acc_*` 会映射到 `mm21_acc_*`。

## Critical Pair 诊断规则

如果实验目录中存在 `CriticalPairStats_record.csv`，可以在报告中加入关键 pair 独占分析：

```bash
python experiment_analysis/generate_analysis.py \
  --critical-pair "short_name|experiment_dir_name|sub_exp_id|Display label"
```

- critical pair 指 train set 中同一个 `(a,b)` 同时出现 add target 和 mm21 target。
- `q1_exclusive_rate` / `q2_exclusive_rate` 表示同一个 query 同时赢走 add/mm21 两个 target 的比例。
- `split_rate` 表示 add/mm21 由不同 query 赢，是更符合预期的分化信号。
- 报告应同时展示全训练期聚合统计和最后一个 interval 统计，用来判断独占问题是否持续到训练末期。
- 这个 section 只用于分析 assignment，不改变 best epoch 选择，也不替代整体 accuracy 和 query-wise accuracy。

## Pair Risk 诊断规则

如果实验目录中存在 `PairRiskStats_record.csv`，可以在 repeated-run 报告中加入 pair risk 分析：

```bash
python experiment_analysis/generate_analysis.py \
  --pair-risk "short_name|experiment_dir_name|sub_exp_id|Display label"
```

- `PairRiskStats_record.csv` 基于实际 train split 建立 pair taxonomy。
- `single_add` / `single_mm21` 表示训练集中该 `(a,b)` 只出现一种 operation。健康信号是同一个 query 稳定独占；风险分数为 `mixed_rate + min(q1_only_rate, q2_only_rate)`。
- `dual_distinct` 表示训练集中该 `(a,b)` 同时出现 add 和 mm21，且 target 不同。健康信号是 add/mm21 被不同 query 赢走；风险分数为 `q1_exclusive_rate + q2_exclusive_rate`。
- `special_ambiguous` 表示 `a + b == (a * b) % 21`，语义不可区分，不进入 pair risk 表。
- 当前常用配置先混合 add/mm21 数据集再 `random_split`；若开启 `augment_times`，split 单位是增强后的虚拟样本 index，而不是唯一 pair。

## Data-Pair 可视化规则

data-pair 可视化使用每个实验 best epoch 对应的图片：

- train 页签读取 `TrainingResults/query_operation_train_epoch_<epoch>_add.png` 和 `query_operation_train_epoch_<epoch>_mm21.png`。
- eval 页签读取 `EvalResults/query_operation_eval_epoch_<epoch>_add.png` 和 `query_operation_eval_epoch_<epoch>_mm21.png`。
- 同一实验展示 add set 和 mm21 set 两张图。
- 生成脚本会把这些图片复制到报告子目录的 `assets/` 中，HTML 不直接引用原始实验目录。
- Data-Pair Visualization section 按实验纵向排列，不把多个实验横向并排。
- 每一行只展示一个实验；该行内 add set 和 mm21 set 两张图并排展示。
- 在窄屏幕上，两张图可以自动改为上下排列，避免图片或文字溢出。
- 每张 data-pair 图片都应支持点击放大。放大视图使用当前报告目录内的相对路径图片，并提供关闭方式，例如点击背景、关闭按钮或 Esc。
- 图中每个格子表示一个 `(label_a, label_b)` datapair 在当前 operation set 中的状态：`/` 表示当前 set 缺失，`1` 表示 q1 做对，`2` 表示 q2 做对，`1,2` 表示两个 query 都做对，红色 `×` 表示两个 query 都没做对。
- 若某个 pair 同时满足 add 和 mm21，即 `a + b == (a * b) % 21`，该格子使用淡紫色底色；即使它在当前 set 中缺失，也显示紫底 `/`。

## 页签规则

train / eval 切换是全局状态，不是每个 section 的局部状态。

- 页面只放一个全局切换控件。
- 切换控件应固定在浏览器窗口中，例如右上角，这样滚动到任意位置都能切换 train / eval。
- 切换只改变展示的数据来源，不改变 best epoch 选择规则。
