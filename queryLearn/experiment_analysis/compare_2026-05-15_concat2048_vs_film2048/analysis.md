## 结论

- 这次是更干净的 concat vs FiLM 对比：两个实验都是 fixed q1/q2、unit=2048、5 hidden layers、AdamW lr=3e-4、symm_loss=0.05、sanity_check=False，主要差异是 `condition_mode`。
- 按每个 sub-exp 的 train total_loss 选 best epoch 后，concat2048 的 eval harmonic mean 是 0.425 +/- 0.190，FiLM2048 是 0.313 +/- 0.102。
- concat2048 在 add 和 mm21 上均值都更高：add 是 0.342 +/- 0.164 vs 0.281 +/- 0.106，mm21 是 0.575 +/- 0.225 vs 0.361 +/- 0.113。
- 两者 train harmonic 都很高且接近：concat2048 是 0.947 +/- 0.012，FiLM2048 是 0.940 +/- 0.003。这说明当前主要问题不是训练集拟合能力，而是 query assignment 和 eval 泛化稳定性。

## 细节观察

- concat2048 的 sub2 是这组里最强的 run，eval add=0.474、mm21=0.821、H=0.601；但 sub3 掉到 H=0.223，说明 concat 也不稳定。
- FiLM2048 三个 sub-exp 都没有明显成功 run，train-selected eval H 分别是 0.407、0.327、0.205；FiLM2048 扩到 2048 后，仍没有在非 sanity_check setting 下自然形成稳定优势。
- best-by-train 和 best-by-eval-H 仍有错位。比如 FiLM2048 sub3 的 train-selected epoch 是 25000，但 eval harmonic 最好在 57500，说明 train total_loss 不是泛化最优 selector。

## FiLM2048 sub3 critical pair

- FiLM2048 sub3 的 critical pair CSV 中识别出 147 个关键 pair。关键 pair 指同一个 `(a,b)` 在训练集中同时出现 add target 和 mm21 target。
- 最明显的长期独占 pair 是 `(9,11)` 和 `(11,9)`，它们的 add target 都是 20、mm21 target 都是 15；其中 `(9,11)` 被 q2 独占的长期比例约 0.786，`(11,9)` 约 0.652。
- `(3,15)` 也很明显，add=18、mm21=3，被 q1 独占的长期比例约 0.586。
- 你之前提到的 `(0,7)` / `(7,0)` 确实也有 q2 独占现象，长期 q2 独占率约 0.300，最后一个 interval 仍约 0.364 / 0.342。
- 这支持一个局部版本的 winner-take-all 问题：宏观上不一定有某个 query 吃掉全局，但在某些关键 pair 上，同一个 query 会同时赢走 add 和 mm21 两个 target，破坏 pair-level 分工。

## 下一步

- 继续保留这个 critical pair 诊断，把它作为非 sanity_check 实验的常规 section。
- 下一轮不要只看 overall add/mm21 accuracy，也要看 top dominated critical pairs 是否减少，以及 split_rate 是否上升。
- 如果要改 loss，仍应避免显式告诉模型哪个 target 是 add 或 mm21；更合适的方向是抑制同一 `(a,b)` 的单 query 双 target 独占，而不是直接监督 operation identity。
