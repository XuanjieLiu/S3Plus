## 结论

- 在这组三次重复实验里，concat 的 eval 表现整体好于 FiLM。按每个 sub-exp 的 train total_loss 选 best epoch 后，concat 的 eval harmonic 大约是 0.451、0.601、0.223；FiLM 大约是 0.341、0.271、0.288。
- 这次不是 sanity_check，而是 fixed q1/q2 但不强制 add/mm21 分配的设置。结果说明：FiLM 在 sanity_check fixedQ 下可以打穿，不代表它在自由 query assignment 的普通训练里自然稳定胜出。
- 两个模型都不够稳定，sub-exp 之间差异很大。concat 有一个较好的 sub2，但 sub3 明显掉下去；FiLM 三个 sub-exp 都偏低，没有出现明显成功 run。

## 为什么 concat 这次更好

- 这次 concat 用 unit=2048，FiLM 用 unit=1024。虽然 FiLM 有条件调制机制，但 concat 的主干容量更大；在非 sanity_check 设置里，更大的直接拼接网络可能更容易拟合一部分 add/mm21 映射。
- fixed query 只是固定了两个条件向量，不再像 sanity_check 那样告诉模型 q1 必须做 add、q2 必须做 mm21。FiLM 的优势依赖 query 条件能稳定诱导两条计算路径；如果 assignment 信号不够强，它可能仍会在两个路径之间漂移。
- best-by-train 和 best-by-eval-H 并不总一致。例如 concat sub1 的 train-selected epoch 是 50000，但 eval harmonic 最好在 30000；FiLM 多个 sub-exp 也有类似错位。这说明 train total_loss 作为 selector 有用但不完全可靠。

## 需要谨慎的地方

- concat 和 FiLM 不是完全等容量对比：concat 是 2048 hidden unit，FiLM 是 1024 hidden unit。因此这份报告不能直接证明 concat 结构本身优于 FiLM。
- 三次重复的方差很大，当前更像是在比较“这个训练 recipe 的稳定性”，不是最终模型能力上限。
- eval 的 add_acc 普遍偏低，说明普通训练还没有稳定地把 add 和 mm21 都学好；当前阶段应该优先诊断 query assignment 和 checkpoint selection。

## 下一步验证

- 做等容量对比：FiLM unit=2048 或 concat unit=1024，确认结构差异是否仍存在。
- 在非 sanity_check 设置中加入更明确的 query specialization 诊断或正则，例如鼓励 q1/q2 在 add/mm21 上形成互补，而不是只看 overall。
- 比较两种 checkpoint selector：train total_loss vs eval harmonic，仅用于分析选择，不一定用于训练早停，判断 train loss 是否系统性错过泛化更好的 epoch。
