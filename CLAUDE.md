# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All notebooks are designed to run on **Google Colab** with data stored on **Google Drive**. Local path references (`./data/`, `./model_storage/`) assume the Colab working directory after copying from Drive. There are no build, lint, or test commands — the project is entirely Jupyter notebooks.

Install dependencies:
```bash
pip install pandas polars numpy matplotlib seaborn pybaseball xgboost lightgbm scikit-learn shap scipy tqdm joblib
```

## Pipeline Execution Order

Each notebook depends on the outputs of the previous one. Run in this order:

1. **`data_fetch.ipynb`** — fetches raw Statcast CSVs from pybaseball (one per year: `2021_data.csv` … `2025_data.csv`)
2. **`data_preprocessing.ipynb`** — produces `train_data.csv`, `test_data_2025.csv`, and lookup artifacts (see below)
3. **`stuff_model_regression.ipynb`** — original unified model; kept as a reference/comparison baseline
4. **`stuff_model_separated.ipynb`** — three production LightGBM models (one per pitch category) saved to `./model_storage/`
5. **`2025_stuff_analysis.ipynb`** — loads production models, scores 2025 pitches, generates leaderboards

## Architecture

### Target Variable

The target is **average delta run expectancy** grouped by `(des_new, count, p_throws, stand)` — not raw per-pitch `delta_run_exp`. Averaging over the (outcome × count × matchup) cell removes pitch-outcome noise while preserving the pitch-quality signal. The lookup table is computed from training data only.

### Data Leakage Prevention

`data_preprocessing.ipynb` computes all lookup tables and models **from training data (2021–2024) only**, then applies them to 2025:

- `target_lookup.csv` — average run value by outcome/count/matchup
- `pitcher_stats_lookup.csv` — per-pitcher fastball averages (speed, az, ax, VAA)
- `magnus_model_x.joblib` / `magnus_model_z.joblib` — linear models for SSW proxy residuals

When processing 2025 data, the Magnus models are loaded from disk (not retrained). Pitcher stats for 2025 are computed from 2025 data itself (current-year differentials are standard for Stuff+ models).

### Feature Engineering Order (matters)

Inside `data_preprocessing.ipynb`, the pipeline must follow this sequence:
1. `df_clean()` — standardize outcomes, filter bad pitch types and sensor artifacts
2. `compute_vaa()` — compute VAA before pitcher stats (avg_fastball_vaa depends on it)
3. `calculate_pitcher_stats_lookup()` — fastball baselines from training only
4. `calculate_magnus_model()` — linear regression on Magnus physics features from training only
5. `apply_target_values()` — join target lookup
6. `feature_engineering()` — applies HAA, vaa_diff, SSW residuals, and differentials

### Handedness Mirroring

`ax`, `release_pos_x`, `vx0`, and `pfx_x` are **sign-flipped for LHP** in `feature_engineering()` to put all pitchers in a right-hand pitcher frame. VAA is purely vertical and is **not** mirrored.

### Production Models (stuff_model_separated.ipynb)

Three LightGBM models trained with walk-forward cross-validation (expanding window: 2021→2022, 2021-22→2023, 2021-23→2024):

| Model | Pitch Types |
|---|---|
| `lgb_fastball_model.joblib` | FF, SI, FC, FA |
| `lgb_breakingball_model.joblib` | SL, ST, CU, KC, SV |
| `lgb_offspeed_model.joblib` | CH, FS, FO |

All three models share the same 19 features and are scaled together to a unified 20-80 scouting scale: `stuff_plus = clip(50 − z_score × 10, 20, 80)`. In `2025_stuff_analysis.ipynb`, the mean is re-centered to the 2025 league average each year (50 = that year's average), while `ref_std` is fixed from training to preserve calibrated spread.

### Excluded Pitch Types

`KN`, `EP`, `PO`, `SC`, `UN`, `CS` are excluded before training and scoring — they violate the spin/movement assumptions the model is built on.

## Known Issues

- **Scale compression**: `ref_std` is computed at the pitch level (~0.025), but is used to scale pitcher-level averages. This compresses the effective leaderboard range to roughly 47–58 instead of 20–80.
- **FA pitch type**: The generic fastball tag (`FA`) scores ~35 on average — likely a data classification artifact given its tiny sample size (3,768 training pitches vs 936k FF).
- **Low ICC**: Intraclass correlation is especially low for FF (0.08), meaning most Stuff+ variance is within-pitcher noise rather than between-pitcher signal. This is a fundamental consequence of pitch-by-pitch scoring and is not fixed in the current implementation.
- **Arsenal score formula** in `2025_stuff_analysis.ipynb` is ad hoc (`weighted_avg_stuff × 0.6 + quality_pitches × 3 + pitch_types × 2`) and over-rewards pitch variety.
