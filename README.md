# intrinsic-symm-melody

This repo keeps the transition and inducement approaches separate but parallel.
Transition is the current result path; inducement is the older result path.

## Layout

- `run_training.py`, `run_evaluation.py`: shared entrypoints.
- `trainer.py`, `tester.py`, `model/`: shared/base ISymm code.
- `model/transition/`: transition-specific trainer/tester/model/loss/config.
- `model/inducement/`: inducement-specific trainer/tester/model/loss/config.
- `experiments/transition/`: transition scripts, logs, CSVs, plots, and final outputs.
- `experiments/inducement/`: inducement scripts, logs, CSVs, plots, and old outputs.
- `experiments/other_runs/`: older loose scripts/results that are not part of the transition/inducement split.

## Current Result

The final transition plot set is the `1119` set:

- `experiments/transition/results/final_1119/figures/transition_performance_plot_1119.png`
- `experiments/transition/results/final_1119/figures/transition_performance_plot_x_1119.png`
- `experiments/transition/results/final_1119/figures/transition_performance_plot_z_1119.png`

The CSV files used by that final plot are in:

- `experiments/transition/results/final_1119/source_csv/`

Run historical shell and plotting scripts from the repo root so their relative
paths resolve correctly.
