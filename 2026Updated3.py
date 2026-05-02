import nflreadpy as nfl
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import webbrowser
import os

# ==========================================
# 1. DOWNLOAD DATA
# ==========================================
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
print("Downloading NFL weekly and player data...")
df_weekly = nfl.load_player_stats(years).to_pandas()

df_players = nfl.load_players().to_pandas()
player_info = df_players[['gsis_id', 'birth_date', 'headshot']].copy()
player_info['birth_date'] = pd.to_datetime(player_info['birth_date'])

valid_positions = ['QB', 'RB', 'WR', 'TE']
df_weekly = df_weekly[df_weekly['position'].isin(valid_positions)].copy()

print("Downloading play-by-play data to calculate Defensive EPA Matchup Difficulty...")
df_pbp = nfl.load_pbp(years).to_pandas()

pass_epa = df_pbp[['season', 'defteam', 'passer_player_id', 'epa']].dropna()
pass_epa = pass_epa.rename(columns={'passer_player_id': 'player_id'})

rush_epa = df_pbp[['season', 'defteam', 'rusher_player_id', 'epa']].dropna()
rush_epa = rush_epa.rename(columns={'rusher_player_id': 'player_id'})

rec_epa = df_pbp[['season', 'defteam', 'receiver_player_id', 'epa']].dropna()
rec_epa = rec_epa.rename(columns={'receiver_player_id': 'player_id'})

epa_combined = pd.concat([pass_epa, rush_epa, rec_epa], ignore_index=True)
epa_combined = epa_combined.merge(df_players[['gsis_id', 'position']], left_on='player_id', right_on='gsis_id', how='inner')
epa_combined = epa_combined[epa_combined['position'].isin(valid_positions)]

def_epa_pos = epa_combined.groupby(['season', 'defteam', 'position'], as_index=False).agg(
    def_epa_allowed=('epa', 'mean')
)

df_weekly = df_weekly.merge(
    def_epa_pos,
    left_on=['season', 'opponent_team', 'position'],
    right_on=['season', 'defteam', 'position'],
    how='left'
)
df_weekly['def_epa_allowed'] = df_weekly['def_epa_allowed'].fillna(0)

# ==========================================
# 2. AGGREGATE DATA TO SEASON TOTALS
# ==========================================
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
    games_played=('week', 'count'),
    avg_matchup_epa=('def_epa_allowed', 'mean')
)
yearly_df['total_tds'] = yearly_df['passing_tds_sum'] + yearly_df['rushing_tds_sum'] + yearly_df['receiving_tds_sum']
yearly_df.drop(columns=['passing_tds_sum', 'rushing_tds_sum', 'receiving_tds_sum'], inplace=True)

# ==========================================
# 3. ENGINEER NEW FEATURES
# ==========================================
yearly_df = yearly_df.merge(player_info, left_on='player_id', right_on='gsis_id', how='left')
yearly_df['season_start_date'] = pd.to_datetime(yearly_df['season'].astype(str) + '-09-01')
yearly_df['age'] = (yearly_df['season_start_date'] - yearly_df['birth_date']).dt.days / 365.25
yearly_df.drop(columns=['gsis_id', 'birth_date', 'season_start_date'], inplace=True)

games = yearly_df['games_played'].replace(0, np.nan)

# --- NEW: Calculate target variable (Points Per Game) ---
yearly_df['pts_per_game_ppr'] = yearly_df['total_pts_ppr'] / games
yearly_df['targets_per_game'] = yearly_df['total_targets'] / games
yearly_df['carries_per_game'] = yearly_df['total_carries'] / games
yearly_df['tds_per_game'] = yearly_df['total_tds'] / games

# Cap efficiency metrics to prevent wild outliers from 1-game wonders
yearly_df['yards_per_carry'] = (yearly_df['total_rushing_yards'] / yearly_df['total_carries'].replace(0, np.nan)).clip(lower=-5, upper=10)
yearly_df['yards_per_target'] = (yearly_df['total_receiving_yards'] / yearly_df['total_targets'].replace(0, np.nan)).clip(lower=-5, upper=20)

headshots = yearly_df.pop('headshot')
yearly_df.fillna(0, inplace=True)
yearly_df['headshot'] = headshots

# ==========================================
# 4. CREATE "PREVIOUS YEAR" FEATURES
# ==========================================
yearly_df = yearly_df.sort_values(by=['player_id', 'season'])
# --- NEW: Added 'pts_per_game_ppr' to shifted features ---
features_to_shift = [
    'pts_per_game_ppr', 'age', 'targets_per_game', 'carries_per_game',
    'tds_per_game', 'yards_per_carry', 'yards_per_target', 'games_played',
    'avg_matchup_epa'
]

for feature in features_to_shift:
    yearly_df[f'prev_{feature}'] = yearly_df.groupby('player_id')[feature].shift(1)

# ==========================================
# 5. CREATE TRAINING SET
# ==========================================
model_df = yearly_df.dropna(subset=['prev_pts_per_game_ppr']).copy()

# --- NEW: Filter out "noise" in training (must have played at least 6 games previously) ---
model_df = model_df[model_df['prev_games_played'] >= 6]

features = [
    'position', 'prev_age', 'prev_games_played', 'prev_pts_per_game_ppr',
    'prev_targets_per_game', 'prev_carries_per_game', 'prev_tds_per_game',
    'prev_yards_per_carry', 'prev_yards_per_target', 'prev_avg_matchup_epa'
]

# --- NEW: Target is now Per-Game scoring, not Total scoring ---
target = 'pts_per_game_ppr'

train_df = model_df[model_df['season'] < 2025]
X_train, y_train = train_df[features], train_df[target]

# ==========================================
# 6. BUILD AND TRAIN THE MODEL
# ==========================================
numeric_features = [col for col in features if col not in ['position']]
categorical_features = ['position']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
])

print("Training the advanced model with per-game targets...")
rf_pipeline.fit(X_train, y_train)

# =======================================================
# 7. PREDICT 2026 AND CALCULATE WEEKLY AVERAGES
# =======================================================
print("Generating 2026 projections...")
stats_2025 = yearly_df[yearly_df['season'] == 2025].copy()

# Only project for players who actually established a baseline in 2025
stats_2025 = stats_2025[stats_2025['games_played'] >= 4].copy()

for feature in features_to_shift:
    stats_2025[f'prev_{feature}'] = stats_2025[feature]

# Because the model is trained on PPG, the direct output IS the weekly projection!
projected_weekly_2026 = rf_pipeline.predict(stats_2025[features])
stats_2025['Projected_Weekly_PPR'] = projected_weekly_2026

# Calculate Non-PPR by subtracting last year's receptions per game
games_2025 = stats_2025['games_played'].replace(0, np.nan)
receptions_per_game_2025 = stats_2025['total_receptions'] / games_2025
stats_2025['Projected_Weekly_NonPPR'] = stats_2025['Projected_Weekly_PPR'] - receptions_per_game_2025

stats_2025['Projected_Weekly_PPR'] = stats_2025['Projected_Weekly_PPR'].fillna(0)
stats_2025['Projected_Weekly_NonPPR'] = stats_2025['Projected_Weekly_NonPPR'].fillna(0)

# ==========================================
# 8. GENERATE HTML DASHBOARD & DISPLAY
# ==========================================
print("Generating HTML Visuals...")

html_content = """
<html>
<head>
    <title>2026 Fantasy Football Projections</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 20px; text-align: center;}
        h1 { color: #333; }
        h2 { color: #0056b3; margin-top: 40px; border-bottom: 2px solid #0056b3; padding-bottom: 10px; display: inline-block;}
        table { margin: 0 auto; border-collapse: collapse; width: 60%; background-color: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        th, td { padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }
        th { background-color: #0056b3; color: white; }
        img { border-radius: 50%; object-fit: cover; }
    </style>
</head>
<body>
    <h1>🏈 2026 Fantasy Football Projections (Per-Game EPA Model) 🏈</h1>
"""

for pos in ['QB', 'RB', 'WR', 'TE']:
    position_df = stats_2025[stats_2025['position'] == pos]
    top_10 = position_df.sort_values(by='Projected_Weekly_PPR', ascending=False).head(10).copy()
    
    top_10['Projected_Weekly_PPR'] = top_10['Projected_Weekly_PPR'].round(2)
    top_10['Projected_Weekly_NonPPR'] = top_10['Projected_Weekly_NonPPR'].round(2)
    
    top_10['Image'] = top_10['headshot'].apply(
        lambda x: f'<img src="{x}" width="60" height="60">' if pd.notnull(x) and x != 0 else '👤'
    )
    
    display_df = top_10[['Image', 'player_name', 'Projected_Weekly_PPR', 'Projected_Weekly_NonPPR']]
    display_df.columns = ['Headshot', 'Player Name', 'Weekly PPR', 'Weekly Standard']

    html_table = display_df.to_html(escape=False, index=False)
    
    html_content += f"<h2>Top 10 Projected {pos}s</h2>\n"
    html_content += html_table + "\n"

html_content += """
</body>
</html>
"""

file_path = os.path.abspath("projections_2026.html")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(html_content)

print(f"\nDone! Dashboard saved to {file_path}")
print("Opening in your web browser...")
webbrowser.open(f"file://{file_path}")