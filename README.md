# Stuff+ Model: Baseball Pitch Quality Analytics

A comprehensive machine learning project that predicts pitch quality using MLB Statcast data. The model calculates "Stuff+" ratings (on a 20-80 scouting scale) based on pitch physical characteristics like velocity, spin rate, and movement.

---

## Overview

This project builds a predictive model to evaluate pitch quality by analyzing how pitch characteristics affect expected run value. The model uses gradient boosting algorithms (XGBoost and LightGBM) trained on 2.8+ million pitches from 2021-2024 MLB seasons to generate Stuff+ ratings that measure pitch effectiveness independent of outcomes.

**Key Results:**
- **Model Performance**: RMSE of 0.2156 (LightGBM) on cross-validation
- **Temporal Stability**: R² of 0.77-0.80 year-over-year correlation
- **Sample Efficiency**: Reliable ratings with as few as 50 pitches per pitch type

---

## Project Structure

### Notebooks

1. **[data_fetch.ipynb](data_fetch.ipynb)** - Data Collection
   - Fetches pitch-by-pitch data from MLB Statcast using pybaseball
   - Covers 2021-2025 regular seasons (~750k pitches per year)
   - Exports raw CSV files for processing

2. **[data_preprocessing.ipynb](data_preprocessing.ipynb)** - Feature Engineering
   - Cleans and standardizes pitch descriptions and events
   - Calculates pitcher baseline stats (fastball velocity, movement)
   - Engineers differential features (speed_diff, az_diff, ax_diff)
   - Mirrors horizontal metrics for left-handed pitchers
   - Prevents data leakage by calculating lookup tables from training data only

3. **[stuff_model_regression.ipynb](stuff_model_regression.ipynb)** - Model Training & Validation
   - Trains XGBoost and LightGBM models with 5-fold cross-validation
   - Analyzes feature importance using SHAP values
   - Validates temporal stability across multiple seasons
   - Performs split-half reliability analysis for sample size recommendations
   - Converts predictions to 20-80 scouting scale (Stuff+)

4. **[2025_stuff_analysis.ipynb](2025_stuff_analysis.ipynb)** - 2025 Season Analysis
   - Generates Stuff+ leaderboards for starting and relief pitchers
   - Analyzes individual pitch types (fastballs, sliders, curveballs, etc.)
   - Evaluates pitcher arsenals and multi-pitch effectiveness
   - Examines velocity vs. Stuff+, spin rate correlations, and platoon splits
   - Provides consistency metrics and pitch usage analysis

---

## Key Features

### Model Features (13 total)
- **Pitch Characteristics**: release_speed, release_spin_rate, spin_axis, release_extension
- **Movement**: az (vertical break), ax (horizontal break)
- **Release Point**: release_pos_x, release_pos_z
- **Differentials**: speed_diff, az_diff, ax_diff (relative to pitcher's fastball)
- **Context**: stand (batter handedness), p_throws (pitcher handedness)

### Target Variable
- **Average Delta Run Expectancy**: Expected change in run value for pitch outcome (des_new, count, matchup)

---

## Installation & Setup

### Requirements
```bash
pip install pandas polars numpy matplotlib seaborn
pip install pybaseball xgboost lightgbm scikit-learn
pip install shap scipy tqdm joblib
```

### Running the Project
1. **Fetch Data**: Run [data_fetch.ipynb](data_fetch.ipynb) to download Statcast data
2. **Preprocess**: Run [data_preprocessing.ipynb](data_preprocessing.ipynb) to clean and engineer features
3. **Train Model**: Run [stuff_model_regression.ipynb](stuff_model_regression.ipynb) to build and validate models
4. **Analyze**: Run [2025_stuff_analysis.ipynb](2025_stuff_analysis.ipynb) to generate insights

---

## Model Validation

### Cross-Validation Results
- **LightGBM**: 0.2156 RMSE (1000 rounds, learning_rate=0.02)
- **XGBoost**: 0.2202 RMSE (500 rounds, learning_rate=0.05)

### Temporal Stability
Year-over-year correlations for pitcher-pitch combinations (min. 200 pitches):
- 2021→2022: R² = 0.796
- 2022→2023: R² = 0.794
- 2023→2024: R² = 0.767

### Sample Size Recommendations
- **50 pitches**: Reliability > 0.95 for fastballs and sliders
- **200 pitches**: Reliability > 0.98 (recommended for public leaderboards)

---

## Sample Insights (2025 Season)

### Top Pitchers by Stuff+
- **Starting Pitchers**: Hunter Greene (61.4), Trey Yesavage (61.5), Jacob Misiorowski (58.3)
- **Relief Pitchers**: Emmanuel Clase (66.6), Ryan Helsley (65.4), Mason Montgomery (64.4)

### Pitch Type Rankings (by average Stuff+)
1. Sweeper (ST): 54.0
2. Slider (SL): 52.7
3. Knuckle Curve (KC): 51.8
4. Sinker (SI): 50.9
5. Splitter (FS): 50.0

---

## Built With

- **Python** - Core programming language
- **pybaseball** - MLB Statcast data API
- **Polars & Pandas** - Data manipulation
- **XGBoost & LightGBM** - Gradient boosting models
- **Scikit-learn** - Model evaluation and utilities
- **SHAP** - Feature importance analysis
- **Matplotlib & Seaborn** - Data visualization

---

## Acknowledgments

- MLB Statcast for providing comprehensive pitch tracking data
- pybaseball library for easy data access
- Inspired by public Stuff+ models from Eno Sarris, Driveline Baseball, and baseball analytics community