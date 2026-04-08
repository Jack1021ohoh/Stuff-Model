# Stuff+ Model: Baseball Pitch Quality Analytics

A machine learning project that predicts pitch quality using MLB Statcast data. The model calculates "Stuff+" ratings on a 20-80 scouting scale based on pitch physical characteristics like velocity, spin rate, movement, and approach angles.

---

## Overview

This project builds three separate LightGBM models — one per pitch category (Fastball, Breaking Ball, Offspeed) — trained on 2.8+ million pitches from the 2021–2024 MLB seasons. Each pitch is scored on a unified 20-80 scouting scale where 50 = league average for that season.

**Key Results:**
- **Model Performance**: Walk-forward CV RMSE of 0.217 (Fastball), 0.208 (Breaking Ball), 0.213 (Offspeed)
- **Temporal Stability**: Mean R² of 0.59–0.69 year-over-year across pitch categories
- **Sample Efficiency**: Most pitch types reach reliable ratings (r ≥ 0.80) at 50–150 pitches

---

## Project Structure

### Notebooks

1. **[data_fetch.ipynb](data_fetch.ipynb)** - Data Collection
   - Fetches pitch-by-pitch data from MLB Statcast using pybaseball
   - Covers 2021–2025 regular seasons (~750k pitches per year)
   - Exports raw CSV files for processing

2. **[data_preprocessing.ipynb](data_preprocessing.ipynb)** - Feature Engineering
   - Cleans and standardizes pitch descriptions and events
   - Computes Vertical/Horizontal Approach Angles (VAA, HAA) from raw Statcast kinematics
   - Trains a linear Magnus baseline model to derive Seam-Shifted Wake (SSW) proxy features
   - Calculates pitcher fastball baselines and differential features
   - Mirrors horizontal metrics for left-handed pitchers
   - Prevents data leakage by computing all lookup tables from training data only

3. **[stuff_model_regression.ipynb](stuff_model_regression.ipynb)** - Unified Model (Reference)
   - Original single LightGBM model trained on all pitch types
   - Kept as a comparison baseline for the separated models

4. **[stuff_model_separated.ipynb](stuff_model_separated.ipynb)** - Production Models
   - Trains three independent LightGBM models (Fastball, Breaking Ball, Offspeed)
   - Uses walk-forward cross-validation (expanding window, 3 folds)
   - Pools all three models' predictions onto a single unified 20-80 scale
   - Includes feature importance, YoY stability, and split-half reliability analysis

5. **[2025_stuff_analysis.ipynb](2025_stuff_analysis.ipynb)** - 2025 Season Analysis
   - Generates Stuff+ leaderboards for starting and relief pitchers
   - Analyzes individual pitch types and pitcher arsenals
   - Examines velocity vs. Stuff+, spin rate correlations, and platoon splits

### Running the Project

1. **Fetch Data**: Run [data_fetch.ipynb](data_fetch.ipynb)
2. **Preprocess**: Run [data_preprocessing.ipynb](data_preprocessing.ipynb)
3. **Train Models**: Run [stuff_model_separated.ipynb](stuff_model_separated.ipynb)
4. **Analyze**: Run [2025_stuff_analysis.ipynb](2025_stuff_analysis.ipynb)

---

## Model Features (19 total)

- **Pitch Characteristics**: release_speed, release_spin_rate, spin_axis, release_extension
- **Movement**: az (vertical break), ax (horizontal break)
- **Release Point**: release_pos_x, release_pos_z
- **Differentials**: speed_diff, az_diff, ax_diff (relative to pitcher's fastball)
- **Approach Angles**: VAA, HAA, vaa_diff (VAA relative to pitcher's fastball)
- **SSW Proxy**: ssw_x, ssw_z, ssw_magnitude (residuals from Magnus movement baseline)
- **Context**: stand (batter handedness), p_throws (pitcher handedness)

### Target Variable
**Average Delta Run Expectancy** grouped by (outcome, count, pitcher handedness, batter handedness) — averaged to reduce pitch-outcome noise while preserving pitch-quality signal.

### Pitch Category Models

| Model | Pitch Types |
|---|---|
| **Fastball** | FF, SI, FC, FA |
| **Breaking Ball** | SL, ST, CU, KC, SV |
| **Offspeed** | CH, FS, FO |

---

## Model Validation

### Walk-Forward CV Results (Separated Models)

| Category | Fold 1 (→2022) | Fold 2 (→2023) | Fold 3 (→2024) | Mean RMSE |
|---|---|---|---|---|
| Fastball | 0.2146 | 0.2194 | 0.2158 | 0.2166 |
| Breaking Ball | 0.2050 | 0.2112 | 0.2079 | 0.2080 |
| Offspeed | 0.2122 | 0.2141 | 0.2115 | 0.2126 |

### Year-over-Year Stability (min. 200 pitches per pitcher-pitch combination)

| Category | 21→22 | 22→23 | 23→24 | Mean R² |
|---|---|---|---|---|
| Fastball | 0.55 | 0.59 | 0.64 | 0.59 |
| Breaking Ball | 0.73 | 0.65 | 0.68 | 0.69 |
| Offspeed | 0.51 | 0.73 | 0.56 | 0.60 |

### Sample Size for Reliable Ratings (split-half reliability)

| Pitch Type | Pitches for r ≥ 0.70 | Pitches for r ≥ 0.80 |
|---|---|---|
| FF, SI | 100 | 150 |
| FC, SL, ST, CU, KC, CH | 50 | 50–100 |

---

## Sample Insights (2025 Season)

### Top Pitchers by Stuff+
- **Starting Pitchers**: Hunter Greene (54.0), Framber Valdez (53.8), Jacob deGrom (53.6)
- **Relief Pitchers**: Trevor Megill (55.0), Emmanuel Clase (54.9), Ryan Helsley (54.4)

### Best Individual Pitches
- Paul Skenes' Splitter (57.9), Erik Miller's Sinker (57.2), Aaron Ashby's Curveball (56.9)

### Pitch Type Rankings (by 2025 average Stuff+)
1. Sweeper (ST): 51.0
2. Slider (SL): 50.8
3. Sinker (SI): 50.5
4. Knuckle Curve (KC): 50.2
5. Splitter (FS): 50.2

---

## Installation

```bash
pip install pandas polars numpy matplotlib seaborn
pip install pybaseball xgboost lightgbm scikit-learn
pip install shap scipy tqdm joblib
```

---

## Built With

- **Python** - Core programming language
- **pybaseball** - MLB Statcast data API
- **Polars & Pandas** - Data manipulation
- **LightGBM & XGBoost** - Gradient boosting models
- **Scikit-learn** - Model evaluation and utilities
- **SHAP** - Feature importance analysis
- **Matplotlib & Seaborn** - Data visualization

---

## Acknowledgments

- MLB Statcast for providing comprehensive pitch tracking data
- pybaseball library for easy data access
- Inspired by public Stuff+ models from Eno Sarris, Driveline Baseball, and the baseball analytics community
