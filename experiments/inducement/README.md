# Inducement Experiments

This is the older inducement result path, kept parallel to transition.

- `scripts/`: old train/eval shell scripts.
- `results/`: old CSV and figure outputs.
- `logs/`: old inducement checkpoints and logs.
- `notebooks/`: old inducement notebooks.

Inducement method code and config live in `model/inducement/`. The shared
entrypoints still import that code when a config uses `method: "ISymm_Induced"`.
