## 结论

- 第二个实验 film-fixedQ 明显优于第一个 film-fc。在按 train total_loss 选择的 best epoch 上，film-fc 的 eval add_acc=0.974、mm21_acc=0.758、harmonic=0.852；film-fixedQ 的 eval add_acc=1.000、mm21_acc=1.000、harmonic=1.000。
- 两个实验都已经学出了清晰的 query 分工：q1 主要负责 add，q2 主要负责 mm21。差别在于 film-fc 的 q2 在 eval mm21 上只有 0.758，而 film-fixedQ 的 q2 在 eval mm21 上达到 1.000。
- 这说明 FiLM-OperNet 的容量大概率足够；目前真正敏感的是 query 初始化和优化稳定性，而不是 FiLM 条件化本身做不到 add/mm21 双任务。

## 为什么第二个实验更好

- 第二个实验固定了更分离的 query 初始化：q1=[1,-1]，q2=[-1,1]。这让 FiLM 的 gamma/beta 从训练早期就能收到方向相反的条件信号，减少两个 query 被 operNet 当成相近条件处理的风险。
- learning_rate 从 1e-3 降到 3e-4，同时改用 AdamW、weight_decay=1e-4 和 grad_clip_norm=1.0。这些改动都在降低训练震荡和后期漂移；第一个实验 memo 里已经观察到 eval 在 10k-15k 后退化，因此稳定优化很可能是关键因素。
- symm_loss_scalar 从 0.2 降到 0.05。sanity check 的首要目标是让 operNet 对不同 query 走出 add/mm21 两条计算路径；过强的 symmetry regularization 可能会压扁两条路径的差异，降低 q2/mm21 的泛化。
- eval/checkpoint 间隔从 5000 改到 2500，并且更早保存 checkpoint。这本身不提升模型能力，但能更细地捕捉最佳节点，避免错过早期稳定区间。

## 需要谨慎的地方

- 第二个实验一次改了多个因素，所以当前结果不能证明哪一个参数单独贡献最大。最可能的组合是 fixed separated query 提供语义分离，较小学习率和 AdamW/clip 负责把这个分离稳定保持住。
- 现在的 fixedQ sanity check 已经接近打穿，但这仍然是“给定 q1/q2 语义”的设置，还没有证明 query learner 可以自己学出 add/mm21 语义。

## 下一步验证

- 做 ablation：只改 fixed query；只改学习率和 AdamW/clip；只改 symm_loss_scalar。这样能拆出到底是初始化、优化稳定性，还是 symmetry 权重在起主导作用。
- 在 fixedQ 配置稳定后，逐步放开 query 学习：先从 separated init 开始 train_queries=True，再测试随机 init 是否也能收敛到 add/mm21 分工。
- 保留 query-wise 指标作为主要诊断：理想形态是 q1 的 add 高、mm21 近 chance，q2 的 mm21 高、add 近 0，同时 overall add/mm21 都高。
