import nflreadpy as nfl
import pandas as pd
import numpy as np

# 1. Fetch Weekly Data (2019 to 2025)
years =[2019, 2020, 2021, 2022, 2023, 2024, 2025]
print("Downloading NFL data using the new nflreadpy...")

# nflreadpy uses load_player_stats instead of import_weekly_data
df = nfl.load_player_stats(years).to_pandas()

# 2. Filter for Offensive Skill Positions
valid_positions =['QB', 'RB', 'WR', 'TE']
df = df[df['position'].isin(valid_positions)].copy()

# 3. Feature Engineering: Rolling 3-Week Averages
# Sort chronologically to ensure rolling windows are accurate
df = df.sort_values(by=['player_id', 'season', 'week'])

# Calculate previous 3 weeks' average fantasy points (shift(1) prevents predicting the future)
df['rolling_3wk_pts'] = df.groupby('player_id')['fantasy_points_ppr'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

# Calculate previous 3 weeks' average targets and carries (usage metrics)
if 'targets' in df.columns and 'carries' in df.columns:
    df['rolling_3wk_targets'] = df.groupby('player_id')['targets'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['rolling_3wk_carries'] = df.groupby('player_id')['carries'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

# 4. Clean up the dataset
# Drop rows where we don't have rolling data (e.g., a player's first game ever)
df = df.dropna(subset=['rolling_3wk_pts', 'fantasy_points_ppr'])

print(f"Total examples ready for modeling: {len(df)}")






# Define Features (X) and Labels (y)
# Note: You can add more features here like 'opponent_team', 'home_away', etc.
features =['position', 'rolling_3wk_pts', 'rolling_3wk_targets', 'rolling_3wk_carries']
target = 'fantasy_points_ppr'

# Split based on season
train_df = df[df['season'].isin([2019, 2020, 2021, 2022, 2023])]
val_df   = df[df['season'] == 2024]
test_df  = df[df['season'] == 2025]

X_train, y_train = train_df[features], train_df[target]
X_val, y_val     = val_df[features], val_df[target]
X_test, y_test   = test_df[features], test_df[target]

print(f"Training size: {len(X_train)} | Validation size: {len(X_val)} | Test size: {len(X_test)}")


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Preprocessing: One-Hot Encode 'position', Standardize numerical features
numeric_features =['rolling_3wk_pts', 'rolling_3wk_targets', 'rolling_3wk_carries']
categorical_features = ['position']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- BASELINE MODEL: Multiple Linear Regression ---
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_val)

print("--- Baseline Linear Regression (Validation) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, lr_preds)):.2f}")
print(f"MAE:  {mean_absolute_error(y_val, lr_preds):.2f}\n")


# --- ADVANCED MODEL: Random Forest Regressor ---
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_val)

print("--- Random Forest Regressor (Validation) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, rf_preds)):.2f}")
print(f"MAE:  {mean_absolute_error(y_val, rf_preds):.2f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Extract the trained model and preprocessor
fitted_rf = rf_pipeline.named_steps['model']
fitted_preprocessor = rf_pipeline.named_steps['preprocessor']

# Get the feature names after one-hot encoding
cat_features_encoded = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_features_encoded)

# Get feature importances
importances = fitted_rf.feature_importances_

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=all_feature_names, palette="viridis")
plt.title("Random Forest Feature Importances for Fantasy Football Predictions")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()