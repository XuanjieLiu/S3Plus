## Core conclusions

- This is a train-only full-train comparison. None of the three experiments has Eval_record.txt, so the report measures train-set fitting and query specialization, not generalization.
- The fair aggregate uses only common sub-exps 1 and 2. The two mul-based experiments also have sub-exp 3, but the add-based experiment only has sub-exp 1 and 2, so sub-exp 3 is excluded from the three-way aggregate.
- Best overall setting is mul-based SPS with symm_loss_scalar=0.0005. Across sub-exp 1 and 2, it reaches harmonic=0.982 +/- 0.000, add_acc=0.976 +/- 0.001, mm21_acc=0.987 +/- 0.000, and total_loss=0.00323 +/- 0.00005.
- The 5/30 mul-based baseline with symm_loss_scalar=0.005 is still strong, but worse and less stable: harmonic=0.972 +/- 0.004, add_acc=0.963 +/- 0.007, mm21_acc=0.980 +/- 0.002, total_loss=0.01072 +/- 0.00273.
- The add-based SPS variant is much weaker: harmonic=0.849 +/- 0.014, add_acc=0.807 +/- 0.021, mm21_acc=0.895 +/- 0.006, total_loss=0.06599 +/- 0.00821.

## What changed

- Baseline 5/30: concat OperNet, fixed q1/q2, mul-based SPS checkpoint, symm_loss_scalar=0.005.
- Add-based variant 6/5: same query and OperNet setup, but SPS initialization changes to the add/triple-set checkpoint while symm_loss_scalar remains 0.005.
- Lower-symm variant 6/5: same mul-based SPS as baseline, but symm_loss_scalar drops from 0.005 to 0.0005.
- All three configs currently have use_label_codebook=False, so this report does not evaluate the new 21-code label-codebook path.

## Interpretation

- The SPS initialization matters a lot. In this mixed add/mm21 setting, the mul-based SPS geometry gives OperNet a much easier target space than the add-based SPS geometry.
- Reducing symm regularization by 10x helps substantially. The lower-symm mul-based run improves harmonic by about +0.010 over the 5/30 mul-based baseline and cuts total_loss by about 70%.
- The lower-symm setting also stabilizes the aggregate: its harmonic std across the two shared sub-exps is about 0.0004, compared with about 0.0042 for the stronger-symm baseline and about 0.0139 for add-based SPS.
- q1/q2 semantics are allowed to swap by seed. This is normal here because q1 and q2 are fixed but symmetric labels. What matters is whether one query specializes toward add and the other toward mm21 within a sub-exp, not whether q1 always means add.

## Pair-risk reading

- Add-based SPS shows severe dual-op failure. Its last-interval dual pair risk has many pairs with risk_score=1.0, meaning the same query wins both add and mm21 for those pairs instead of splitting.
- The mul-based baseline has much better accuracy but still leaves notable dual risks, for example sub-exp 1 has pair (7,0) with last-interval risk about 0.596 and split_rate about 0.016.
- The lower-symm mul-based run fixes many single-op competition cases, but it does not eliminate all dual-op same-query dominance. In sub-exp 1 pair (15,3) has risk_score=1.0, and in sub-exp 2 pairs (0,7) and (7,0) have risk_score=1.0.
- Therefore high overall accuracy does not mean the query semantics are perfectly disentangled. The next bottleneck is a small set of stubborn dual_distinct pairs.

## Recommended next checks

- Treat mul-based SPS with symm_loss_scalar=0.0005 as the current best baseline for full-train fitting.
- Run the same best setting with operator.use_label_codebook=True to test whether the 21-prototype quantizer makes the target space cleaner.
- Add a small symm sweep around 0.0005, such as 0, 0.0001, 0.0005, and 0.001, because the 0.005 setting looks too strong for this objective.
- Inspect Data-Pair Visualization for exp1 and exp2 tabs, especially the top dual-risk pairs, before adding any loss term. The remaining errors appear localized rather than global.
