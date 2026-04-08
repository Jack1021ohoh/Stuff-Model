# Stuff+ Model Improvement Plan V2

Three targeted improvements to address target variable noise, scale compression, and a data artifact in pitch classification.

---

## 1. Exclude FA (Generic Fastball)

**Problem:** FA (generic fastball) averages ~35 Stuff+ — a full 15 points below league average. This is not a real signal. FA is Statcast's catch-all tag for unclassified fastballs, assigned inconsistently and in tiny volume (3,768 training pitches out of 2.8M). The model has insufficient data to learn FA reliably, and the scores it produces are a classification artifact, not pitch quality.

**Change:** Add `'FA'` to `EXCLUDED_PITCH_TYPES` in `data_preprocessing.ipynb`, alongside `KN`, `EP`, `PO`, `SC`, `UN`, `CS`. Remove `FA` from the `PITCH_CATEGORIES['Fastball']` list in `stuff_model_separated.ipynb` and `2025_stuff_analysis.ipynb`.

**Files affected:** `data_preprocessing.ipynb`, `stuff_model_separated.ipynb`, `2025_stuff_analysis.ipynb`

---

## 2. Improve the Target Variable with Expected Contact Quality (xRV)

**Problem:** The current target averages actual `delta_run_exp` by (outcome, count, handedness matchup). For non-contact outcomes (ball, called strike, whiff, foul), this is clean — those outcomes have no luck component. But for contact (`hit_into_play`), the actual run value depends on fielding and park factors. A hard-hit line drive caught by a diving outfielder gets the same run value as a weak popup, even though the pitcher who allowed the liner deserved worse.

**Goal:** Replace actual contact outcomes with **expected contact run value** based on exit velocity and launch angle. This removes fielding luck and park factors, leaving only the quality of contact the pitch induced.

### Implementation

**Step 1: Add required Statcast columns to `data_fetch.ipynb`**

Ensure `estimated_woba_using_speedangle`, `launch_speed`, and `launch_angle` are retained in the raw data fetch. These are available in Statcast for all batted balls.

**Step 2: Fit an xRV regression in `data_preprocessing.ipynb`**

Train a simple linear regression (from training data only) mapping `estimated_woba_using_speedangle` → `delta_run_exp` on contact pitches. This keeps everything in the same run-value units as the rest of the target and avoids manual scale conversions.

```python
def calculate_xrv_contact_model(train_df: pl.DataFrame):
    """
    Fit a linear regression from estimated_woba_using_speedangle → delta_run_exp
    on contact pitches from training data only.
    Keeps expected contact run value in the same units as non-contact delta_run_exp.
    """
    contact_df = (
        train_df
        .filter(pl.col('des_new') == 'hit_into_play')
        .select(['estimated_woba_using_speedangle', 'delta_run_exp'])
        .drop_nulls()
        .to_pandas()
    )

    X = contact_df[['estimated_woba_using_speedangle']].values
    y = contact_df['delta_run_exp'].values

    model = LinearRegression().fit(X, y)
    print(f"xRV contact model R²: {model.score(X, y):.4f}")
    print(f"  Intercept: {model.intercept_:.4f}, Slope: {model.coef_[0]:.4f}")
    return model
```

Save with `joblib.dump(xrv_contact_model, './data/xrv_contact_model.joblib')`.

**Step 3: Modify `calculate_target_lookup()` to use xRV for contact**

Instead of averaging actual `delta_run_exp` for all pitches uniformly, apply the xRV model to replace contact run values before averaging:

```python
def calculate_target_lookup(train_df: pl.DataFrame, xrv_contact_model) -> pl.DataFrame:
    """
    Build the run-value lookup table using:
    - Actual delta_run_exp for non-contact outcomes (ball, strike, whiff, foul, HBP)
    - Expected delta_run_exp (from xRV model) for contact outcomes
    """
    contact_mask = pl.col('des_new') == 'hit_into_play'

    # Predict xRV for contact rows
    contact_df = train_df.filter(contact_mask)
    X_contact = contact_df.select('estimated_woba_using_speedangle').drop_nulls().to_pandas()
    xrv_preds = xrv_contact_model.predict(X_contact[['estimated_woba_using_speedangle']].values)

    contact_with_xrv = contact_df.with_columns(
        pl.Series('delta_run_exp_adj', xrv_preds)
    )

    # Non-contact rows: use actual delta_run_exp unchanged
    non_contact_with_xrv = (
        train_df
        .filter(~contact_mask)
        .with_columns(pl.col('delta_run_exp').alias('delta_run_exp_adj'))
    )

    combined = pl.concat([contact_with_xrv, non_contact_with_xrv], how='diagonal')

    target_lookup = combined.group_by(['des_new', 'count', 'p_throws', 'stand']).agg([
        pl.col('delta_run_exp_adj').mean().alias('target')
    ])
    return target_lookup
```

**Step 4: Apply to 2025 data**

Load `xrv_contact_model.joblib` (do NOT refit on 2025 data) and pass it into `calculate_target_lookup()` when processing `test_data_2025.csv`. The preprocessing pipeline is otherwise unchanged.

**What this does NOT change:** The grouping structure `(des_new, count, p_throws, stand)` is preserved. The target is still an averaged expected run value — only the run value used for contact rows is now expectation-based rather than outcome-based.

**Files affected:** `data_fetch.ipynb` (column retention), `data_preprocessing.ipynb` (new model + modified target lookup)

---

## 3. Intra-Pitch-Type Scaling

**Problem:** The current model pools all predictions together to compute a single `ref_mean` and `ref_std`:

```python
ref_mean = all_data['y_pred'].mean()  # mix of FF, SL, CH, CU, ...
ref_std  = all_data['y_pred'].std()
```

Because different pitch types have different prediction distributions (breaking balls have smaller spread than fastballs), pooling compresses elite pitches of tighter-distribution types. A slider with a Z-score of -2.5 relative to other sliders gets diluted when measured against the pooled distribution. This is the root cause of the leaderboard ceiling at ~58.

**Fix:** Scale each pitch type against its **own** distribution. A slider is graded against sliders, a fastball against fastballs.

### Implementation

**Step 1: Compute per-pitch-type scaling parameters from training data in `stuff_model_separated.ipynb`**

```python
# After generating all training predictions in all_data['y_pred']
scaler_by_pitch_type = {}

for pitch_type in all_data['pitch_type'].unique():
    pt_preds = all_data[all_data['pitch_type'] == pitch_type]['y_pred']
    scaler_by_pitch_type[pitch_type] = {
        'ref_mean': pt_preds.mean(),
        'ref_std':  pt_preds.std(),
    }

joblib.dump(scaler_by_pitch_type, './model_storage/stuff_plus_per_pt_scaler.joblib')
```

**Step 2: Replace `calculate_stuff_plus()` with a per-pitch-type version**

```python
def calculate_stuff_plus_per_pt(df, scaler_by_pitch_type, recenter_to_year=True):
    """
    Scale each pitch type against its own distribution.
    If recenter_to_year=True, re-center ref_mean to the current year's
    per-pitch-type average (so 50 = this year's average for that pitch type),
    while preserving the training ref_std for calibrated spread.
    """
    df = df.copy()
    df['stuff_plus'] = np.nan

    for pitch_type, scaler in scaler_by_pitch_type.items():
        mask = df['pitch_type'] == pitch_type
        if mask.sum() == 0:
            continue

        ref_std  = scaler['ref_std']
        ref_mean = df.loc[mask, 'y_pred'].mean() if recenter_to_year else scaler['ref_mean']

        z = (df.loc[mask, 'y_pred'] - ref_mean) / ref_std
        df.loc[mask, 'stuff_plus'] = np.clip(50 - z * 10, 20, 80)

    return df
```

**Step 3: Update `2025_stuff_analysis.ipynb`**

Replace the call to the old scaler with the new per-pitch-type scaler:

```python
scaler_by_pitch_type = joblib.load('./model_storage/stuff_plus_per_pt_scaler.joblib')
data_2025 = calculate_stuff_plus_per_pt(data_2025, scaler_by_pitch_type, recenter_to_year=True)
```

### What this changes

| | Before | After |
|---|---|---|
| Scaling reference | All pitch types pooled | Each pitch type separately |
| Elite slider Z-score | ~-0.8 (diluted by pooled std) | ~-2.5 (relative to sliders only) |
| Leaderboard ceiling | ~58 | ~70+ for genuinely elite pitches |
| Cross-pitch-type comparisons | Compressed | Meaningful — 60 SL = 60 FF (both 1σ above their type's average) |

**Files affected:** `stuff_model_separated.ipynb` (scaler computation), `2025_stuff_analysis.ipynb` (scoring call)

---

## Implementation Order

| Step | Task | Dependency |
|---|---|---|
| 1 | Exclude FA from `EXCLUDED_PITCH_TYPES` and `PITCH_CATEGORIES` | None |
| 2 | Add `estimated_woba_using_speedangle` to data fetch | None |
| 3 | Fit xRV contact model, update `calculate_target_lookup()` | Step 2 |
| 4 | Re-run `data_preprocessing.ipynb` to regenerate `train_data.csv` | Step 3 |
| 5 | Re-train separated models | Step 4 |
| 6 | Compute per-pitch-type scaler | Step 5 |
| 7 | Update `2025_stuff_analysis.ipynb` with new scaler | Step 6 |

Steps 1 and 2 have no dependencies and can be done first in either order.
