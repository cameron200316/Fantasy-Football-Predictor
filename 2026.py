import nflreadpy as nfl
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# 1. Download Data
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
print("Downloading NFL data for season predictions...")
df = nfl.load_player_stats(years).to_pandas()

valid_positions = ['QB', 'RB', 'WR', 'TE']
df = df[df['position'].isin(valid_positions)].copy()

# 2. Aggregate Data to Season Totals
# We need receptions to calculate the difference between PPR and Non-PPR
yearly_df = df.groupby(['player_id', 'player_name', 'position', 'season'], as_index=False).agg(
    total_pts_ppr=('fantasy_points_ppr', 'sum'),
    total_targets=('targets', 'sum'),
    total_carries=('carries', 'sum'),
    total_receptions=('receptions', 'sum'), # Added to calculate Non-PPR
    games_played=('week', 'count')
)

# 3. Create "Previous Year" Features for the model
yearly_df = yearly_df.sort_values(by=['player_id', 'season'])
yearly_df['prev_pts_ppr'] = yearly_df.groupby('player_id')['total_pts_ppr'].shift(1)
yearly_df['prev_targets'] = yearly_df.groupby('player_id')['total_targets'].shift(1)
yearly_df['prev_carries'] = yearly_df.groupby('player_id')['total_carries'].shift(1)
yearly_df['prev_games'] = yearly_df.groupby('player_id')['games_played'].shift(1)

# 4. Create Training Set
model_df = yearly_df.dropna(subset=['prev_pts_ppr']).copy()

# Define features and the target (we predict the SEASON TOTAL)
features = ['position', 'prev_pts_ppr', 'prev_targets', 'prev_carries', 'prev_games']
target = 'total_pts_ppr'

train_df = model_df[model_df['season'] < 2025]
X_train, y_train = train_df[features], train_df[target]

# 5. Build and Train the Random Forest Pipeline
numeric_features = ['prev_pts_ppr', 'prev_targets', 'prev_carries', 'prev_games']
categorical_features = ['position']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # n_jobs=-1 uses all available CPU cores to speed up training
    ('model', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
])

print("Training the model on historical season data...")
rf_pipeline.fit(X_train, y_train)

# =======================================================
# 6. PREDICT 2026 AND CALCULATE WEEKLY AVERAGES
# =======================================================
print("Generating 2026 projections...")
# Get all player stats from the most recent completed season (2025)
stats_2025 = yearly_df[yearly_df['season'] == 2025].copy()

# Rename 2025 stats to match the feature names the model was trained on
stats_2025['prev_pts_ppr'] = stats_2025['total_pts_ppr']
stats_2025['prev_targets'] = stats_2025['total_targets']
stats_2025['prev_carries'] = stats_2025['total_carries']
stats_2025['prev_games'] = stats_2025['games_played']

# Predict the SEASON TOTAL for 2026 using 2025 data
projected_total_2026 = rf_pipeline.predict(stats_2025[features])
stats_2025['Projected_2026_Total_PPR'] = projected_total_2026

# --- Convert Season Totals to Weekly Averages ---
# Avoid division by zero for players who played 0 games
# We use 2025 games played as the projected number of games for 2026
stats_2025['Projected_Weekly_PPR'] = stats_2025['Projected_2026_Total_PPR'] / 18

# Estimate Non-PPR score by subtracting their receptions per game from 2025
receptions_per_game_2025 = stats_2025['total_receptions'] / 18
stats_2025['Projected_Weekly_NonPPR'] = stats_2025['Projected_Weekly_PPR'] - receptions_per_game_2025

# Clean up any potential NaN values from division by zero
stats_2025.fillna(0, inplace=True)


# ==========================================
# 7. DISPLAY THE RESULTS
# ==========================================
print("\n")
# Loop through each position and print the top 10 ranked by PPR weekly average
for pos in ['QB', 'RB', 'WR', 'TE']:
    # Filter projections for players in the current position who played a meaningful number of games
    position_df = stats_2025[(stats_2025['position'] == pos) & (stats_2025['games_played'] > 5)]

    # Sort by projected PPR weekly average
    top_10 = position_df.sort_values(by='Projected_Weekly_PPR', ascending=False).head(10)

    # Prepare the columns for a clean display
    display_df = top_10[['player_name', 'Projected_Weekly_PPR', 'Projected_Weekly_NonPPR']].round(2)

    print(f"🏈 TOP 10 PROJECTED {pos}s FOR 2026 (WEEKLY AVG) 🏈")
    print("-" * 60)
    print(display_df.to_string(index=False))
    print("\n")