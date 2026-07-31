## Core conclusions

- This is a train-only full-train comparison. None of the three experiments has Eval_record.txt, so the report measures train-set fitting and query specialization, not generalization.
- The fair aggregate uses only common sub-exps 1 and 2. The 8d noSymm experiment also has sub-exp 3, but it is excluded from the three-way aggregate.
- Best overall setting is the 8d AddBased SPS with no symmetry regularization. Across sub-exp 1 and 2, it reaches harmonic=0.917 +/- 0.009, add_acc=0.891 +/- 0.011, mm21_acc=0.944 +/- 0.006, and total_loss=0.00524 +/- 0.00077.
- The 8d AddBased SPS with symm_loss_scalar=0.005 is worse but stable: harmonic=0.882 +/- 0.001, add_acc=0.845 +/- 0.002, mm21_acc=0.921 +/- 0.001, and total_loss=0.01125 +/- 0.00004.
- The older 2d AddBased SPS with symm_loss_scalar=0.005 is weakest in this comparison: harmonic=0.849 +/- 0.014, add_acc=0.807 +/- 0.021, mm21_acc=0.895 +/- 0.006, and total_loss=0.06599 +/- 0.00821.

## What changed

- 6/5 2d AddBased: concat OperNet, fixed q1/q2, add/triple-set SPS checkpoint with edim1, symm_loss_scalar=0.005.
- 6/26 8d AddBased noSymm: same concat fixed-query OperNet setup, but SPS changes to the newer edim4 8d AddBased checkpoint and is_symm=False.
- 6/28 8d AddBased symm: same newer 8d AddBased SPS as 6/26, but is_symm=True with symm_loss_scalar=0.005.
- All three are full-train settings with train_data_ratio=1.0 and use_label_codebook=False.

## Interpretation

- Moving from the older 2d AddBased SPS to the newer 8d AddBased SPS is a large improvement for OperNet fitting. The noSymm 8d run cuts total_loss by about 92% relative to the 2d symm run and improves harmonic by about +0.068.
- In this add-based SPS family, symm_loss_scalar=0.005 appears too strong. Adding it to the 8d SPS raises total_loss from about 0.00524 to 0.01125 and lowers harmonic from about 0.917 to 0.882.
- The 8d symm run is very stable across the two seeds, but the stability is around a lower-accuracy solution. This looks more like regularization-induced underfitting than a useful specialization improvement.
- q1/q2 labels are allowed to swap by seed. What matters is whether one query specializes toward add and the other toward mm21 within a sub-exp, not whether q1 always means add.

## Pair-risk reading

- The 8d noSymm run has the lowest last-interval dual-pair risk among the three: mean dual risk is about 0.084, versus about 0.265 for 2d symm and about 0.241 for 8d symm.
- The 8d noSymm run still has stubborn dual_distinct failures. Some pairs reach risk_score=1.0 in the last interval, so high aggregate accuracy does not mean perfect query disentanglement.
- The 8d symm run introduces more single-op query competition than noSymm. Last-interval single-pair risk mean rises from about 0.0003 in noSymm to about 0.0171 in symm, with top single-pair risk around 0.292.
- The older 2d symm run has near-zero single-op competition, but many severe dual-op same-query dominance cases. That means it can keep single-operation assignments stable while still failing to split add/mm21 for shared (a,b) pairs.

## Recommended next checks

- Treat 8d AddBased SPS with noSymm as the current AddBased baseline for this concat fixed-query setting.
- If symmetry is revisited, sweep smaller values such as 0.0001, 0.0005, and 0.001 instead of jumping to 0.005.
- Inspect the Data-Pair Visualization for exp1 and exp2 tabs, especially the top dual-risk pairs in the 8d noSymm run, before adding a new loss term.
- Compare this best AddBased baseline against the best MulBased baseline from the prior report to separate SPS geometry effects from query-conditioning effects.
