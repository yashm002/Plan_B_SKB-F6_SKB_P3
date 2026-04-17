# ============================================================
# CROP YIELD PREDICTION PIPELINE
# Ready to run in Google Colab
# ============================================================
# SETUP: Run this cell first in Colab to install dependencies
# !pip install pandas numpy scikit-learn matplotlib seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')



print("=" * 60)
print("STEP 1: Loading Datasets")
print("=" * 60)


DATASET1_PATH = "dataset1.csv"  
DATASET2_PATH = "dataset2.csv"   



try:
    df1 = pd.read_csv(DATASET1_PATH)
    df2 = pd.read_csv(DATASET2_PATH)
    print(f"Dataset 1 loaded: {df1.shape[0]} rows, {df1.shape[1]} columns")
    print(f"Dataset 2 loaded: {df2.shape[0]} rows, {df2.shape[1]} columns")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    print("Generating synthetic sample data for demonstration...")

    # ---- SYNTHETIC DATA (only used if files are missing) ----
    np.random.seed(42)
    n1, n2 = 5000, 3000
    states = ['Punjab', 'Haryana', 'UP', 'MP', 'Maharashtra', 'Karnataka', 'TamilNadu']
    crops  = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Soybean']
    seasons = ['Kharif', 'Rabi', 'Zaid', 'Whole Year']

    df1 = pd.DataFrame({
        'State_Name':    np.random.choice(states, n1),
        'District_Name': ['District_' + str(i) for i in np.random.randint(1, 50, n1)],
        'Crop_Year':     np.random.randint(2000, 2020, n1),
        'Season':        np.random.choice(seasons, n1),
        'Crop':          np.random.choice(crops, n1),
        'Area':          np.random.uniform(100, 5000, n1).round(2),
        'Production':    np.random.uniform(500, 50000, n1).round(2),
    })

    df2 = pd.DataFrame({
        'Crop':             np.random.choice(crops, n2),
        'Crop_Year':        np.random.randint(2005, 2020, n2),
        'Season':           np.random.choice(seasons, n2),
        'State':            np.random.choice(states, n2),
        'Area':             np.random.uniform(100, 5000, n2).round(2),
        'Production':       np.random.uniform(500, 50000, n2).round(2),
        'Annual_Rainfall':  np.random.uniform(300, 2000, n2).round(2),
        'Fertilizer':       np.random.uniform(50, 500, n2).round(2),
        'Pesticide':        np.random.uniform(1, 50, n2).round(2),
        'Yield':            np.random.uniform(1, 15, n2).round(4),
    })
    print("Synthetic data created successfully.")

print(f"\nDataset 1 columns: {list(df1.columns)}")
print(f"Dataset 2 columns: {list(df2.columns)}")

# ============================================================
# STEP 2: CLEAN BOTH DATASETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Cleaning Datasets")
print("=" * 60)

def clean_dataset(df, name):
    """Cleans a dataframe: strips whitespace, fixes dtypes, drops full-duplicate rows."""
    print(f"\n--- Cleaning {name} ---")
    original_shape = df.shape

    # Strip leading/trailing whitespace from string columns
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Standardise string columns to Title Case for consistent merging
    for col in str_cols:
        df[col] = df[col].str.title()

    # Ensure Crop_Year is integer
    if 'Crop_Year' in df.columns:
        df['Crop_Year'] = pd.to_numeric(df['Crop_Year'], errors='coerce')
        df.dropna(subset=['Crop_Year'], inplace=True)
        df['Crop_Year'] = df['Crop_Year'].astype(int)

    # Drop rows that are entirely duplicate
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Removed {before - len(df)} fully duplicate rows")
    print(f"  Shape: {original_shape} → {df.shape}")
    return df

df1 = clean_dataset(df1, "Dataset 1")
df2 = clean_dataset(df2, "Dataset 2")

# ============================================================
# STEP 3: RENAME COLUMNS FOR MERGE COMPATIBILITY
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Renaming Columns")
print("=" * 60)

# Dataset 1 uses 'State_Name'; rename to 'State' so both datasets match
if 'State_Name' in df1.columns:
    df1.rename(columns={'State_Name': 'State'}, inplace=True)
    print("  df1: Renamed 'State_Name' → 'State'")

# Standardise the Annual_Rainfall column name in Dataset 2 (handles truncated name)
rainfall_candidates = [c for c in df2.columns if 'annual' in c.lower() or 'rainfall' in c.lower()]
if rainfall_candidates and rainfall_candidates[0] != 'Annual_Rainfall':
    df2.rename(columns={rainfall_candidates[0]: 'Annual_Rainfall'}, inplace=True)
    print(f"  df2: Renamed '{rainfall_candidates[0]}' → 'Annual_Rainfall'")

print(f"\n  df1 columns now: {list(df1.columns)}")
print(f"  df2 columns now: {list(df2.columns)}")

# ============================================================
# STEP 4: MERGE DATASETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Merging on [Crop, Crop_Year, State]")
print("=" * 60)

MERGE_KEYS = ['Crop', 'Crop_Year', 'State']

# Verify merge keys exist in both dataframes
for key in MERGE_KEYS:
    assert key in df1.columns, f"Merge key '{key}' missing from Dataset 1!"
    assert key in df2.columns, f"Merge key '{key}' missing from Dataset 2!"

# Outer merge to retain as much data as possible
merged = pd.merge(df1, df2, on=MERGE_KEYS, how='outer', suffixes=('_df1', '_df2'))
print(f"  Merged shape: {merged.shape}")

# ---- Consolidate duplicate columns created by the merge ----
# Area: prefer df2 value; fall back to df1
if 'Area_df1' in merged.columns and 'Area_df2' in merged.columns:
    merged['Area'] = merged['Area_df2'].combine_first(merged['Area_df1'])
    merged.drop(columns=['Area_df1', 'Area_df2'], inplace=True)

# Production: prefer df2 value; fall back to df1
if 'Production_df1' in merged.columns and 'Production_df2' in merged.columns:
    merged['Production'] = merged['Production_df2'].combine_first(merged['Production_df1'])
    merged.drop(columns=['Production_df1', 'Production_df2'], inplace=True)

# Season: prefer df2 if both exist
if 'Season_df1' in merged.columns and 'Season_df2' in merged.columns:
    merged['Season'] = merged['Season_df2'].combine_first(merged['Season_df1'])
    merged.drop(columns=['Season_df1', 'Season_df2'], inplace=True)

print(f"  Columns after consolidation: {list(merged.columns)}")

# ============================================================
# STEP 5: POST-MERGE CLEANING
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Post-Merge Cleaning")
print("=" * 60)

# 5a. Remove duplicates
before = len(merged)
merged.drop_duplicates(inplace=True)
print(f"  Removed {before - len(merged)} duplicate rows after merge")

# 5b. Calculate Yield if missing (Yield = Production / Area)
if 'Yield' not in merged.columns:
    merged['Yield'] = np.nan

missing_yield_mask = merged['Yield'].isna() & merged['Production'].notna() & merged['Area'].notna() & (merged['Area'] > 0)
merged.loc[missing_yield_mask, 'Yield'] = merged.loc[missing_yield_mask, 'Production'] / merged.loc[missing_yield_mask, 'Area']
print(f"  Calculated Yield for {missing_yield_mask.sum()} rows from Production / Area")

# 5c. Handle remaining missing values
print(f"\n  Missing values before imputation:\n{merged.isnull().sum()[merged.isnull().sum() > 0]}")

# Numeric columns → fill with median (robust to outliers)
numeric_cols = merged.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if merged[col].isna().sum() > 0:
        median_val = merged[col].median()
        merged[col].fillna(median_val, inplace=True)

# Categorical columns → fill with mode
cat_cols = merged.select_dtypes(include='object').columns
for col in cat_cols:
    if merged[col].isna().sum() > 0:
        mode_val = merged[col].mode()[0]
        merged[col].fillna(mode_val, inplace=True)

print(f"\n  Missing values after imputation: {merged.isnull().sum().sum()}")
print(f"  Final merged shape: {merged.shape}")

# ============================================================
# STEP 6: PREPARE FINAL FEATURE SET
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Preparing Final Feature Set")
print("=" * 60)

FEATURES = ['State', 'Crop', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
TARGET   = 'Yield'

# Keep only required columns (drop any that don't exist gracefully)
available = [c for c in FEATURES + [TARGET] if c in merged.columns]
missing_cols = set(FEATURES + [TARGET]) - set(available)
if missing_cols:
    print(f"  WARNING: Columns not found and will be skipped: {missing_cols}")

df_model = merged[available].copy()

# Remove rows where Yield is still missing or non-positive
df_model = df_model[df_model[TARGET].notna() & (df_model[TARGET] > 0)]

# Remove extreme Yield outliers (beyond 99th percentile)
upper_cap = df_model[TARGET].quantile(0.99)
df_model = df_model[df_model[TARGET] <= upper_cap]

print(f"  Model dataset shape: {df_model.shape}")
print(f"  Features: {FEATURES}")
print(f"  Target : {TARGET}")

# Label-encode categorical features
le_dict = {}
cat_features = [c for c in ['State', 'Crop', 'Season'] if c in df_model.columns]
for col in cat_features:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    le_dict[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique classes")

# ============================================================
# STEP 7: TRAIN / TEST SPLIT & MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Training RandomForestRegressor")
print("=" * 60)

feature_cols = [c for c in FEATURES if c in df_model.columns]
X = df_model[feature_cols]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1        # Use all available CPU cores
)
model.fit(X_train, y_train)
print("  Model trained successfully!")

# ============================================================
# STEP 8: MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"  MAE  (Mean Absolute Error) : {mae:.4f}")
print(f"  RMSE (Root Mean Sq. Error) : {rmse:.4f}")
print(f"  R²   (R-Squared)           : {r2:.4f}")

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n  Feature Importances:\n{feat_imp.round(4)}")

# ============================================================
# STEP 9: SCENARIO SIMULATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Scenario Simulation")
print("=" * 60)

# Define scenario multipliers
SCENARIOS = {
    'Normal':   {'Annual_Rainfall': 1.0, 'Fertilizer': 1.0},
    'Drought':  {'Annual_Rainfall': 0.7, 'Fertilizer': 1.0},
    'Flood':    {'Annual_Rainfall': 1.5, 'Fertilizer': 1.0},
    'Heatwave': {'Annual_Rainfall': 0.9, 'Fertilizer': 1.0},
    'Extreme':  {'Annual_Rainfall': 0.6, 'Fertilizer': 0.8},
}

def apply_scenario(data: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Apply a climate scenario to the dataset by scaling
    Annual_Rainfall and Fertilizer columns.

    Parameters
    ----------
    data     : pd.DataFrame — Feature dataset (must contain the model's feature columns)
    scenario : str          — One of 'Normal', 'Drought', 'Flood', 'Heatwave', 'Extreme'

    Returns
    -------
    pd.DataFrame — Modified copy of the data with scenario multipliers applied
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIOS.keys())}")

    modified = data.copy()
    multipliers = SCENARIOS[scenario]

    for col, factor in multipliers.items():
        if col in modified.columns:
            modified[col] = modified[col] * factor

    return modified


# Use the test set as the base for scenario simulation
base_data = X_test.copy()

scenario_results = {}
print(f"\n  {'Scenario':<12} {'Avg Predicted Yield':>22}")
print("  " + "-" * 36)

for scenario_name in SCENARIOS:
    scenario_data  = apply_scenario(base_data, scenario_name)
    predicted_yield = model.predict(scenario_data)
    avg_yield = predicted_yield.mean()
    scenario_results[scenario_name] = avg_yield
    print(f"  {scenario_name:<12} {avg_yield:>22.4f}")

# ============================================================
# STEP 10: PLOT BAR CHART OF SCENARIO YIELDS
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Plotting Scenario Yield Comparison")
print("=" * 60)

scenario_names  = list(scenario_results.keys())
scenario_yields = list(scenario_results.values())

# Colour palette: green for normal/flood, amber for heatwave, red for drought/extreme
colors = {
    'Normal':   '#4CAF50',
    'Flood':    '#2196F3',
    'Heatwave': '#FF9800',
    'Drought':  '#F44336',
    'Extreme':  '#9C27B0',
}
bar_colors = [colors[s] for s in scenario_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Crop Yield Prediction: Scenario Analysis', fontsize=16, fontweight='bold', y=1.02)

# --- Bar chart: Average Yield per Scenario ---
ax1 = axes[0]
bars = ax1.bar(scenario_names, scenario_yields, color=bar_colors, edgecolor='black', linewidth=0.8)
ax1.set_title('Average Predicted Yield by Climate Scenario', fontsize=12)
ax1.set_xlabel('Scenario', fontsize=11)
ax1.set_ylabel('Average Predicted Yield', fontsize=11)
ax1.set_ylim(0, max(scenario_yields) * 1.2)

# Annotate bars with values
for bar, val in zip(bars, scenario_yields):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(scenario_yields) * 0.02,
        f'{val:.2f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

ax1.tick_params(axis='x', rotation=15)
ax1.grid(axis='y', alpha=0.3)

# --- Bar chart: % Change vs Normal ---
ax2 = axes[1]
normal_yield = scenario_results['Normal']
pct_changes  = {s: ((v - normal_yield) / normal_yield) * 100
                for s, v in scenario_results.items() if s != 'Normal'}
pct_colors   = ['#4CAF50' if v >= 0 else '#F44336' for v in pct_changes.values()]

bars2 = ax2.bar(list(pct_changes.keys()), list(pct_changes.values()),
                color=pct_colors, edgecolor='black', linewidth=0.8)
ax2.axhline(0, color='black', linewidth=1)
ax2.set_title('Yield Change (%) vs Normal Scenario', fontsize=12)
ax2.set_xlabel('Scenario', fontsize=11)
ax2.set_ylabel('Change in Yield (%)', fontsize=11)

for bar, val in zip(bars2, pct_changes.values()):
    ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.2
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        ypos,
        f'{val:+.1f}%',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

ax2.tick_params(axis='x', rotation=15)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('scenario_yield_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Chart saved as 'scenario_yield_comparison.png'")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Final dataset rows  : {len(df_model)}")
print(f"  Model R²            : {r2:.4f}")
print(f"  Model RMSE          : {rmse:.4f}")
print(f"\n  Scenario Avg Yields:")
for s, v in scenario_results.items():
    print(f"    {s:<12}: {v:.4f}")
print("=" * 60)