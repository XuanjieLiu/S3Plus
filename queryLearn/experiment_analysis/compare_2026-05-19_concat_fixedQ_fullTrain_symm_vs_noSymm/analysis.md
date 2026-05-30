## Key Takeaways

- This is a train-only fullTrain comparison: both experiments use `train_data_ratio=1.0`, so the report should be read as fitting and specialization analysis, not generalization analysis.
- `noSymm` is the stronger current baseline for concat fixed-query OperNet. Its final train accuracy is higher and its loss is much lower than `symm`.
- `symm` increases q1/q2 specialization on some seeds, but it also raises oper loss and reduces add/mm21 accuracy. The specialization gain is not consistently reflected as better pair-level split.

## Metric Read

- The report tables use the best train checkpoint that also has query operation images, so the image rows and numerical rows refer to the same epoch.
- In the full training logs, final mean harmonic accuracy is about `0.978` for `noSymm` and about `0.932` for `symm`.
- In the full training logs, final mean oper loss is about `0.0026` for `noSymm` and about `0.0234` for `symm`.
- At the final PairRisk interval, mean dual-distinct pair split rate is about `0.941` for `noSymm` and about `0.891` for `symm`.
- `symm` has higher same-query dominance risk on dual pairs, so its stronger q-wise accuracy gap does not automatically mean cleaner add/mm21 semantic separation.

## Interpretation

- In concat mode, the query only enters as extra input dimensions. With fixed q1/q2, this gives weaker conditional control than FiLM-style modulation.
- The symmetry regularizer appears to push the two query paths apart, but at `symm_loss_scalar=0.05` it over-constrains the shared MLP and hurts exact operator fitting.
- The best checkpoints are not always the final checkpoints. Several noSymm sub-exp runs peak earlier, so checkpoint selection matters for this comparison.

## Recommendation

- Keep `noSymm_fixedQ_fullTrain` as the concat fixed-query baseline.
- Do not use `symm_loss_scalar=0.05` as the default for concat fixed-query experiments.
- If symmetry is still worth testing, try smaller weights such as `0.005` or `0.01`, or turn it on after a warmup phase.
- For stronger query-conditioned computation, prefer the FiLM OperNet line over concat.
