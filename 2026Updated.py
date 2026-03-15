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
print("Downloading NFL weekly and player data...")
df_weekly = nfl.load_player_stats(years).to_pandas()

df_players = nfl.load_players().to_pandas()
player_ages = df_players[['gsis_id', 'birth_date']].copy()
player_ages['birth_date'] = pd.to_datetime(player_ages['birth_date'])

valid_positions = ['QB', 'RB', 'WR', 'TE']
df_weekly = df_weekly[df_weekly['position'].isin(valid_positions)].copy()

# 2. Aggregate Data to Season Totals
yearly_df = df_weekly.groupby(['player_id', 'player_name', 'position', 'season'], as_index=False).agg(
    total_pts_ppr=('fantasy_points_ppr', 'sum'),
    total_targets=('targets', 'sum'),
    total_carries=('carries', 'sum'),
    total_receptions=('receptions', 'sum'),
    total_receiving_yards=('receiving_yards', 'sum'),
    total_rushing_yards=('rushing_yards', 'sum'),
    passing_tds_sum=('passing_tds', 'sum'),
    rushing_tds_sum=('rushing_tds', 'sum'),
    receiving_tds_sum=('receiving_tds', 'sum'),
    games_played=('week', 'count')
)
yearly_df['total_tds'] = yearly_df['passing_tds_sum'] + yearly_df['rushing_tds_sum'] + yearly_df['receiving_tds_sum']
yearly_df.drop(columns=['passing_tds_sum', 'rushing_tds_sum', 'receiving_tds_sum'], inplace=True)


# 3. Engineer New Features (Age, Per-Game, and Efficiency)
yearly_df = yearly_df.merge(player_ages, left_on='player_id', right_on='gsis_id', how='left')
yearly_df['season_start_date'] = pd.to_datetime(yearly_df['season'].astype(str) + '-09-01')
yearly_df['age'] = (yearly_df['season_start_date'] - yearly_df['birth_date']).dt.days / 365.25
yearly_df.drop(columns=['gsis_id', 'birth_date', 'season_start_date'], inplace=True)

games = yearly_df['games_played'].replace(0, np.nan)
yearly_df['targets_per_game'] = yearly_df['total_targets'] / games
yearly_df['carries_per_game'] = yearly_df['total_carries'] / games
yearly_df['tds_per_game'] = yearly_df['total_tds'] / games
yearly_df['yards_per_carry'] = yearly_df['total_rushing_yards'] / yearly_df['total_carries'].replace(0, np.nan)
yearly_df['yards_per_target'] = yearly_df['total_receiving_yards'] / yearly_df['total_targets'].replace(0, np.nan)
yearly_df.fillna(0, inplace=True)

# 4. Create "Previous Year" Features for the model
yearly_df = yearly_df.sort_values(by=['player_id', 'season'])
features_to_shift = [
    'total_pts_ppr', 'age', 'targets_per_game', 'carries_per_game',
    'tds_per_game', 'yards_per_carry', 'yards_per_target', 'games_played'
]
for feature in features_to_shift:
    yearly_df[f'prev_{feature}'] = yearly_df.groupby('player_id')[feature].shift(1)

# 5. Create Training Set
model_df = yearly_df.dropna(subset=['prev_total_pts_ppr']).copy()

features = [
    'position', 'prev_age', 'prev_games_played', 'prev_total_pts_ppr',
    'prev_targets_per_game', 'prev_carries_per_game', 'prev_tds_per_game',
    'prev_yards_per_carry', 'prev_yards_per_target'
]
target = 'total_pts_ppr'

train_df = model_df[model_df['season'] < 2025]
X_train, y_train = train_df[features], train_df[target]

# 6. Build and Train the More Advanced Model
numeric_features = [col for col in features if col not in ['position']]
categorical_features = ['position']

# --- FIXED: Added the column lists to the tuples ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
])

print("Training the advanced model with new features...")
rf_pipeline.fit(X_train, y_train)

# =======================================================
# 7. PREDICT 2026 AND CALCULATE WEEKLY AVERAGES
# =======================================================
print("Generating 2026 projections with the advanced model...")
stats_2025 = yearly_df[yearly_df['season'] == 2025].copy()

for feature in features_to_shift:
    stats_2025[f'prev_{feature}'] = stats_2025[feature]

projected_total_2026 = rf_pipeline.predict(stats_2025[features])
stats_2025['Projected_2026_Total_PPR'] = projected_total_2026

games_2025 = stats_2025['games_played'].replace(0, np.nan)
stats_2025['Projected_Weekly_PPR'] = stats_2025['Projected_2026_Total_PPR'] / games_2025
receptions_per_game_2025 = stats_2025['total_receptions'] / games_2025
stats_2025['Projected_Weekly_NonPPR'] = stats_2025['Projected_Weekly_PPR'] - receptions_per_game_2025
stats_2025.fillna(0, inplace=True)

# ==========================================
# 8. DISPLAY THE RESULTS
# ==========================================
print("\n")
for pos in ['QB', 'RB', 'WR', 'TE']:
    position_df = stats_2025[(stats_2025['position'] == pos) & (stats_2025['games_played'] > 5)]
    top_10 = position_df.sort_values(by='Projected_Weekly_PPR', ascending=False).head(10)
    display_df = top_10[['player_name', 'Projected_Weekly_PPR', 'Projected_Weekly_NonPPR']].round(2)

    print(f"🏈 TOP 10 PROJECTED {pos}s FOR 2026 (WEEKLY AVG) - ADVANCED MODEL 🏈")
    print("-" * 70)
    print(display_df.to_string(index=False))
    print("\n")