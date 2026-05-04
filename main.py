import os
import webbrowser
import numpy as np
import pandas as pd
import nflreadpy as nfl

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def download_data(years: list, valid_positions: list):
    """
    Downloads NFL weekly stats, player info, and play-by-play data.
    Calculates Defensive EPA Matchup Difficulty and merges it into the weekly dataset.
    """
    print("Downloading NFL weekly and player data...")
    df_weekly = nfl.load_player_stats(years).to_pandas()
    df_players = nfl.load_players().to_pandas()

    # Extract player info and format birth dates
    player_info = df_players[['gsis_id', 'birth_date', 'headshot']].copy()
    player_info['birth_date'] = pd.to_datetime(player_info['birth_date'])

    # Filter to valid offensive positions
    df_weekly = df_weekly[df_weekly['position'].isin(valid_positions)].copy()

    print("Downloading play-by-play data to calculate Defensive EPA Matchup Difficulty...")
    df_pbp = nfl.load_pbp(years).to_pandas()

    # Extract passing, rushing, and receiving EPA
    pass_epa = df_pbp[['season', 'defteam', 'passer_player_id', 'epa']].dropna()
    pass_epa = pass_epa.rename(columns={'passer_player_id': 'player_id'})

    rush_epa = df_pbp[['season', 'defteam', 'rusher_player_id', 'epa']].dropna()
    rush_epa = rush_epa.rename(columns={'rusher_player_id': 'player_id'})

    rec_epa = df_pbp[['season', 'defteam', 'receiver_player_id', 'epa']].dropna()
    rec_epa = rec_epa.rename(columns={'receiver_player_id': 'player_id'})

    # Combine EPA data and calculate average allowed per position by defense
    epa_combined = pd.concat([pass_epa, rush_epa, rec_epa], ignore_index=True)
    epa_combined = epa_combined.merge(df_players[['gsis_id', 'position']], left_on='player_id', right_on='gsis_id', how='inner')
    epa_combined = epa_combined[epa_combined['position'].isin(valid_positions)]

    def_epa_pos = epa_combined.groupby(['season', 'defteam', 'position'], as_index=False).agg(
        def_epa_allowed=('epa', 'mean')
    )

    # Merge defensive EPA allowed back into the weekly data
    df_weekly = df_weekly.merge(
        def_epa_pos,
        left_on=['season', 'opponent_team', 'position'],
        right_on=['season', 'defteam', 'position'],
        how='left'
    )
    df_weekly['def_epa_allowed'] = df_weekly['def_epa_allowed'].fillna(0)

    return df_weekly, player_info


def aggregate_season_totals(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Groups weekly game logs to calculate yearly totals for each player.
    """
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
    
    # Consolidate touchdowns
    yearly_df['total_tds'] = yearly_df['passing_tds_sum'] + yearly_df['rushing_tds_sum'] + yearly_df['receiving_tds_sum']
    yearly_df.drop(columns=['passing_tds_sum', 'rushing_tds_sum', 'receiving_tds_sum'], inplace=True)

    return yearly_df

def engineer_features(yearly_df: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame:
    """
    Merges player info, calculates age, per-game averages, and efficiency metrics.
    """
    yearly_df = yearly_df.merge(player_info, left_on='player_id', right_on='gsis_id', how='left')
    
    # Calculate player age relative to the start of the season
    yearly_df['season_start_date'] = pd.to_datetime(yearly_df['season'].astype(str) + '-09-01')
    yearly_df['age'] = (yearly_df['season_start_date'] - yearly_df['birth_date']).dt.days / 365.25
    yearly_df.drop(columns=['gsis_id', 'birth_date', 'season_start_date'], inplace=True)

    # Calculate per-game metrics
    games = yearly_df['games_played'].replace(0, np.nan)
    yearly_df['pts_per_game_ppr'] = yearly_df['total_pts_ppr'] / games
    yearly_df['targets_per_game'] = yearly_df['total_targets'] / games
    yearly_df['carries_per_game'] = yearly_df['total_carries'] / games
    yearly_df['tds_per_game'] = yearly_df['total_tds'] / games

    # Calculate efficiency metrics and clip outliers
    yearly_df['yards_per_carry'] = (yearly_df['total_rushing_yards'] / yearly_df['total_carries'].replace(0, np.nan)).clip(lower=-5, upper=10)
    yearly_df['yards_per_target'] = (yearly_df['total_receiving_yards'] / yearly_df['total_targets'].replace(0, np.nan)).clip(lower=-5, upper=20)

    # Fill NA values carefully so headshot URLs are not replaced by 0
    headshots = yearly_df.pop('headshot')
    yearly_df.fillna(0, inplace=True)
    yearly_df['headshot'] = headshots

    return yearly_df

def create_lagged_features(yearly_df: pd.DataFrame, features_to_shift: list) -> pd.DataFrame:
    """
    Creates 'previous year' metrics by shifting the target features by one season.
    """
    yearly_df = yearly_df.sort_values(by=['player_id', 'season'])
    
    for feature in features_to_shift:
        yearly_df[f'prev_{feature}'] = yearly_df.groupby('player_id')[feature].shift(1)
        
    return yearly_df

def prepare_datasets(yearly_df: pd.DataFrame, features: list, target: str):
    """
    Filters datasets for minimum requirements and splits into Train/Val/Test objects.
    """
    # Filter minimum requirements
    model_df = yearly_df.dropna(subset=['prev_pts_per_game_ppr']).copy()
    model_df = model_df[model_df['prev_games_played'] >= 6]

    # Create Train/Val split (Seasons <= 2024)
    train_val_df = model_df[model_df['season'] <= 2024].copy()
    X_train_val = train_val_df[features]
    y_train_val = train_val_df[target]

    # Use 2024 explicitly as validation fold for GridSearchCV
    test_fold = np.where(train_val_df['season'] == 2024, 0, -1)
    ps = PredefinedSplit(test_fold)

    # Create Test Set (Season == 2025)
    test_df = model_df[model_df['season'] == 2025].copy()
    X_test = test_df[features]
    y_test = test_df[target]

    return X_train_val, y_train_val, ps, X_test, y_test

def train_and_evaluate_models(X_train_val, y_train_val, ps, X_test, y_test, features):
    """
    Builds modeling pipelines, tunes Random Forest, calculates evaluation metrics,
    and extracts feature importance.
    """
    numeric_features = [col for col in features if col != 'position']
    categorical_features = ['position']

    # Preprocessing Pipeline Setup
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- BASELINE MODEL: Linear Regression ---
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    lr_pipeline.fit(X_train_val, y_train_val)
    preds_2025_lr = lr_pipeline.predict(X_test)
    mae_2025_lr = mean_absolute_error(y_test, preds_2025_lr)
    rmse_2025_lr = np.sqrt(mean_squared_error(y_test, preds_2025_lr))

    # --- ADVANCED MODEL: Random Forest ---
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10]
    }

    print("\nSearching for best Random Forest parameters using 2024 as the Validation Set...")
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=ps,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train_val, y_train_val)
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Test RF on 2025
    preds_2025_rf = best_rf_model.predict(X_test)
    mae_2025_rf = mean_absolute_error(y_test, preds_2025_rf)
    rmse_2025_rf = np.sqrt(mean_squared_error(y_test, preds_2025_rf))

    # Compile Evaluation Metrics
    metrics = {
        'lr_mae': mae_2025_lr,
        'lr_rmse': rmse_2025_lr,
        'rf_mae': mae_2025_rf,
        'rf_rmse': rmse_2025_rf
    }

    # Extract Feature Importances
    rf_model_step = best_rf_model.named_steps['model']
    preprocessor_step = best_rf_model.named_steps['preprocessor']

    cat_cols_encoded = preprocessor_step.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_cols_encoded)
    importances = rf_model_step.feature_importances_

    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance Weight': importances
    }).sort_values(by='Importance Weight', ascending=False)

    return best_rf_model, metrics, best_params, feat_imp_df

def generate_2026_projections(best_rf_model, yearly_df: pd.DataFrame, features: list, features_to_shift: list) -> pd.DataFrame:
    """
    Maps 2025 performances to 'prev_' columns to feed into the model to predict 2026.
    """
    print("Generating 2026 projections based on 2025 performances...")
    stats_2025 = yearly_df[yearly_df['season'] == 2025].copy()
    stats_2025 = stats_2025[stats_2025['games_played'] >= 4].copy()

    # Shift features manually for prediction set
    for feature in features_to_shift:
        stats_2025[f'prev_{feature}'] = stats_2025[feature]

    # Predict
    projected_weekly_2026 = best_rf_model.predict(stats_2025[features])
    stats_2025['Projected_Weekly_PPR'] = projected_weekly_2026

    # Calculate Non-PPR Projections
    games_2025 = stats_2025['games_played'].replace(0, np.nan)
    receptions_per_game_2025 = stats_2025['total_receptions'] / games_2025
    stats_2025['Projected_Weekly_NonPPR'] = stats_2025['Projected_Weekly_PPR'] - receptions_per_game_2025

    stats_2025['Projected_Weekly_PPR'] = stats_2025['Projected_Weekly_PPR'].fillna(0)
    stats_2025['Projected_Weekly_NonPPR'] = stats_2025['Projected_Weekly_NonPPR'].fillna(0)

    return stats_2025

def create_html_dashboard(stats_2025: pd.DataFrame, metrics: dict, best_params: dict, feat_imp_df: pd.DataFrame):
    """
    Generates an HTML report visualizing model performance and top positional projections.
    """
    print("Generating HTML Visuals...")

    html_content = f"""
    <html>
    <head>
        <title>2026 Fantasy Football Projections</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 20px; text-align: center;}}
            h1 {{ color: #333; }}
            h2 {{ color: #0056b3; margin-top: 40px; border-bottom: 2px solid #0056b3; padding-bottom: 10px; display: inline-block;}}
            table {{ margin: 0 auto; border-collapse: collapse; width: 60%; background-color: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-bottom: 40px; }}
            th, td {{ padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #0056b3; color: white; }}
            img {{ border-radius: 50%; object-fit: cover; }}
            .feature-table th {{ background-color: #28a745; }}
            .feature-table {{ width: 40%; }}
            .info-box {{ background-color: white; width: 50%; margin: 0 auto 40px auto; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; text-align: left;}}
            .info-box h3 {{ color: #d9534f; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            .info-box span {{ font-weight: bold; color: #333; }}
        </style>
    </head>
    <body>
        <h1>2026 Fantasy Football Projections</h1>
        
        <div class="info-box">
            <h3>Model Performance (Tested on 2025 Data)</h3>
            <p><span>Baseline Linear Regression MAE:</span> {metrics['lr_mae']:.2f} Points Per Game off</p>
            <p><span>Tuned Random Forest MAE:</span> {metrics['rf_mae']:.2f} Points Per Game off</p>
            
            <h3 style="margin-top: 20px;">Chosen Random Forest Hyperparameters</h3>
            <p><span>n_estimators:</span> {best_params['model__n_estimators']} <i>(Number of trees in the forest)</i></p>
            <p><span>max_depth:</span> {best_params['model__max_depth']} <i>(Maximum depth of the trees)</i></p>
            <p><span>min_samples_split:</span> {best_params['model__min_samples_split']} <i>(Min samples required to split an internal node)</i></p>
        </div>
    """

    # Generate Top 10 projections for each position
    for pos in ['QB', 'RB', 'WR', 'TE']:
        position_df = stats_2025[stats_2025['position'] == pos]
        top_10 = position_df.sort_values(by='Projected_Weekly_PPR', ascending=False).head(10).copy()
        
        top_10['Projected_Weekly_PPR'] = top_10['Projected_Weekly_PPR'].round(2)
        top_10['Projected_Weekly_NonPPR'] = top_10['Projected_Weekly_NonPPR'].round(2)
        
        # Format headshot images for HTML
        top_10['Image'] = top_10['headshot'].apply(
            lambda x: f'<img src="{x}" width="60" height="60">' if pd.notnull(x) and x != 0 else '👤'
        )
        
        display_df = top_10[['Image', 'player_name', 'Projected_Weekly_PPR', 'Projected_Weekly_NonPPR']]
        display_df.columns = ['Headshot', 'Player Name', 'Weekly PPR', 'Weekly Standard']

        html_table = display_df.to_html(escape=False, index=False)
        html_content += f"<h2>Top 10 Projected {pos}s</h2>\n" + html_table + "\n"

    # Add Feature Importances
    feat_imp_display = feat_imp_df.copy()
    feat_imp_display['Importance Weight'] = (feat_imp_display['Importance Weight'] * 100).round(2).astype(str) + '%'
    feat_imp_html = feat_imp_display.to_html(index=False, classes="feature-table")

    html_content += """
        <h2>Model Feature Importances</h2>
        <p>This shows which stats the algorithm relied on the most to generate predictions.</p>
    """
    html_content += feat_imp_html + "\n</body>\n</html>"

    file_path = os.path.abspath("projections_2026.html")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"\nDone! Dashboard saved to {file_path}")
    print("Opening in your web browser...")
    webbrowser.open(f"file://{file_path}")


if __name__ == "__main__":
    """
    Main function. Defines variables and runs all logical steps.
    """
    # Define Constants & Configuration
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    valid_positions = ['QB', 'RB', 'WR', 'TE']
    
    features_to_shift = [
        'pts_per_game_ppr', 'age', 'targets_per_game', 'carries_per_game',
        'tds_per_game', 'yards_per_carry', 'yards_per_target', 'games_played',
        'avg_matchup_epa'
    ]
    
    features = [
        'position', 'prev_age', 'prev_games_played', 'prev_pts_per_game_ppr',
        'prev_targets_per_game', 'prev_carries_per_game', 'prev_tds_per_game',
        'prev_yards_per_carry', 'prev_yards_per_target', 'prev_avg_matchup_epa'
    ]
    
    target_var = 'pts_per_game_ppr'

    # Execute Data Pipeline
    df_weekly, player_info = download_data(years, valid_positions)
    yearly_df = aggregate_season_totals(df_weekly)
    yearly_df = engineer_features(yearly_df, player_info)
    yearly_df = create_lagged_features(yearly_df, features_to_shift)

    # ML Preprocessing Pipeline
    X_train_val, y_train_val, ps, X_test, y_test = prepare_datasets(yearly_df, features, target_var)
    
    # Train, Tune, and Evaluate
    best_rf_model, metrics, best_params, feat_imp_df = train_and_evaluate_models(
        X_train_val, y_train_val, ps, X_test, y_test, features
    )

    # Print results to console
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON (2025 TEST SET)")
    print("="*50)
    print("Baseline (Linear Regression):")
    print(f"  - MAE  : {metrics['lr_mae']:.2f} PPG off")
    print(f"  - RMSE : {metrics['lr_rmse']:.2f} PPG off")
    print("\nTuned Random Forest:")
    print(f"  - MAE  : {metrics['rf_mae']:.2f} PPG off")
    print(f"  - RMSE : {metrics['rf_rmse']:.2f} PPG off")
    print("\n RANDOM FOREST CHOSEN PARAMETERS")
    print(f"n_estimators      : {best_params['model__n_estimators']}")
    print(f"max_depth         : {best_params['model__max_depth']}")
    print(f"min_samples_split : {best_params['model__min_samples_split']}")
    print("="*50 + "\n")

    # Generate Output Data & Visualizations
    stats_2025 = generate_2026_projections(best_rf_model, yearly_df, features, features_to_shift)
    create_html_dashboard(stats_2025, metrics, best_params, feat_imp_df)