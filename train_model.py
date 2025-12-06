# -*- coding: utf-8 -*-
"""
IPL Win Predictor - Model Training Script
This script trains the machine learning model and saves it as a pickle file.

IMPORTANT: You need to download 'deliveries.csv' first!
Download from: https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set
Place it in the same folder as this script.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import os
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

def check_files():
    """Check if required CSV files exist"""
    if not os.path.exists('matches.csv'):
        raise FileNotFoundError("ERROR: 'matches.csv' not found in current directory!")
    if not os.path.exists('deliveries.csv'):
        raise FileNotFoundError(
            "ERROR: 'deliveries.csv' not found!\n"
            "Please download it from: https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set\n"
            "Or use: https://data.world/raghu543/ipl-data-till-2017"
        )
    print("SUCCESS: All required CSV files found!")

def load_data():
    """Load and prepare the datasets"""
    print("\nLoading datasets...")
    match = pd.read_csv('matches.csv')
    dlvr = pd.read_csv('deliveries.csv')
    print(f"SUCCESS: Loaded {len(match)} matches and {len(dlvr)} deliveries")
    return match, dlvr

def prepare_data(match, dlvr):
    """Prepare and clean the data"""
    print("\nPreparing data...")

    # Calculate total scores
    total_score_df = dlvr.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]

    # Merge datasets
    match_df = match.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')

    # Current IPL teams
    teams = [
        'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
        'Rajasthan Royals', 'Delhi Capitals', 'Punjab Kings',
        'Sunrisers Hyderabad', 'Lucknow Super Giants', 'Gujarat Titans',
        'Royal Challengers Bengaluru'
    ]

    # Replace old team names with current ones
    match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

    match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

    match_df['team1'] = match_df['team1'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    match_df['team2'] = match_df['team2'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')

    match_df['team1'] = match_df['team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
    match_df['team2'] = match_df['team2'].str.replace('Kings XI Punjab', 'Punjab Kings')

    # Filter only current teams
    match_df = match_df[match_df['team1'].isin(teams)]
    match_df = match_df[match_df['team2'].isin(teams)]

    # Convert match_id to int
    match_df['match_id'] = match_df['match_id'].astype(int)
    dlvr['match_id'] = dlvr['match_id'].astype(int)

    # Merge deliveries with match data
    dlvr_df = match_df.merge(dlvr, on='match_id')

    # Filter only second innings
    dlvr_df = dlvr_df[dlvr_df['inning'] == 2]

    print(f"SUCCESS: Prepared {len(dlvr_df)} delivery records")
    return dlvr_df

def engineer_features(dlvr_df):
    """Create features for the model"""
    print("\nEngineering features...")

    # Current runs scored
    dlvr_df['current_runs'] = dlvr_df.groupby('match_id')['total_runs_y'].cumsum()

    # Runs left to win
    dlvr_df['runs_left'] = dlvr_df['total_runs_x'] - dlvr_df['current_runs']

    # Balls left
    dlvr_df['balls_left'] = 126 - (dlvr_df['over'] * 6 + dlvr_df['ball'])

    # Wickets tracking
    dlvr_df['player_dismissed'] = dlvr_df['player_dismissed'].fillna("0")
    dlvr_df['player_dismissed'] = dlvr_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1)
    dlvr_df['player_dismissed'] = dlvr_df['player_dismissed'].astype(int)

    dlvr_df['wickets'] = dlvr_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
    dlvr_df['wickets_left'] = 10 - dlvr_df['wickets']

    # Cumulative runs
    dlvr_df['cumulative_runs'] = dlvr_df.groupby(['match_id', 'inning'])['total_runs_y'].cumsum()

    # Balls bowled
    dlvr_df['balls_bowled'] = dlvr_df['over'] * 6 + dlvr_df['ball']
    dlvr_df['overs_done'] = dlvr_df['balls_bowled'] / 6

    # Current Run Rate
    dlvr_df['crr'] = dlvr_df.apply(
        lambda row: row['cumulative_runs'] / row['overs_done'] if row['overs_done'] > 0 else 0,
        axis=1
    )

    # Required runs and RRR
    dlvr_df['runs_required'] = dlvr_df['target_runs'] - dlvr_df['cumulative_runs']
    dlvr_df['runs_required'] = dlvr_df['runs_required'].apply(lambda x: x if x > 0 else 0)

    dlvr_df['overs_left'] = (120 - dlvr_df['balls_bowled']) / 6

    dlvr_df['rrr'] = dlvr_df.apply(
        lambda row: row['runs_required'] / row['overs_left']
        if row['overs_left'] > 0 and row['inning'] == 2 else 0,
        axis=1
    )

    # NEW FEATURES FOR BETTER ACCURACY

    # 1. Pressure Index (difference between RRR and CRR)
    dlvr_df['pressure'] = dlvr_df['rrr'] - dlvr_df['crr']

    # 2. Runs per wicket remaining
    dlvr_df['runs_per_wicket'] = dlvr_df.apply(
        lambda row: row['runs_left'] / row['wickets_left'] if row['wickets_left'] > 0 else row['runs_left'] * 10,
        axis=1
    )

    # 3. Recent momentum (runs scored in last 3 overs)
    dlvr_df['recent_runs'] = dlvr_df.groupby('match_id')['total_runs_y'].rolling(window=18, min_periods=1).sum().reset_index(0, drop=True)

    # 4. Run rate difference
    dlvr_df['run_rate_diff'] = dlvr_df['crr'] - dlvr_df['rrr']

    # 5. Match situation (categorical: easy, moderate, tough, very tough)
    def get_situation(row):
        if row['wickets_left'] == 0:
            return 'impossible'
        rpo_required = row['runs_left'] / (row['balls_left'] / 6) if row['balls_left'] > 0 else 0
        if rpo_required <= 6:
            return 'easy'
        elif rpo_required <= 9:
            return 'moderate'
        elif rpo_required <= 12:
            return 'tough'
        else:
            return 'very_tough'

    dlvr_df['situation'] = dlvr_df.apply(get_situation, axis=1)

    # Create result column (1 if batting team won, 0 otherwise)
    dlvr_df['result'] = dlvr_df.apply(
        lambda row: 1 if row['batting_team'] == row['winner'] else 0,
        axis=1
    )

    print("SUCCESS: Features engineered successfully (including advanced features)")
    return dlvr_df

def prepare_final_dataset(dlvr_df):
    """Prepare final dataset for training"""
    print("\nPreparing final dataset...")

    final_df = dlvr_df[[
        'batting_team', 'bowling_team', 'city', 'runs_left',
        'balls_left', 'wickets_left', 'total_runs_x', 'crr', 'rrr',
        'pressure', 'runs_per_wicket', 'recent_runs', 'run_rate_diff', 'situation',
        'result'
    ]]

    # Shuffle
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    # Remove nulls
    final_df.dropna(inplace=True)

    # Remove rows where balls_left is 0
    final_df = final_df[final_df['balls_left'] != 0]

    print(f"SUCCESS: Final dataset: {len(final_df)} records with enhanced features")
    return final_df

def train_model(final_df):
    """Train the machine learning model"""
    print("\nTraining multiple models to find the best one...")

    # Split features and target
    X = final_df.iloc[:, :-1]
    y = final_df.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Create preprocessing pipeline
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first'),
         ['batting_team', 'bowling_team', 'city', 'situation'])
    ], remainder='passthrough')

    # Try multiple models
    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=1)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    from sklearn.metrics import accuracy_score

    print("\nComparing models:")
    print("-" * 50)

    for name, model in models.items():
        # Create pipeline
        pipe = Pipeline(steps=[
            ('preprocessor', trf),
            ('classifier', model)
        ])

        # Train
        pipe.fit(X_train, y_train)

        # Evaluate
        y_pred = pipe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name}: {accuracy * 100:.2f}%")

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipe
            best_name = name

    print("-" * 50)
    print(f"\nBEST MODEL: {best_name}")
    print(f"BEST ACCURACY: {best_accuracy * 100:.2f}%")

    return best_model

def save_model(model, filename='ipl_model.pkl'):
    """Save the trained model"""
    print(f"\nSaving model to '{filename}'...")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"SUCCESS: Model saved successfully!")

def main():
    """Main execution function"""
    print("=" * 60)
    print("IPL WIN PREDICTOR - MODEL TRAINING")
    print("=" * 60)

    try:
        # Step 1: Check files
        check_files()

        # Step 2: Load data
        match, dlvr = load_data()

        # Step 3: Prepare data
        dlvr_df = prepare_data(match, dlvr)

        # Step 4: Engineer features
        dlvr_df = engineer_features(dlvr_df)

        # Step 5: Prepare final dataset
        final_df = prepare_final_dataset(dlvr_df)

        # Step 6: Train model
        model = train_model(final_df)

        # Step 7: Save model
        save_model(model)

        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now run: streamlit run app.py")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo download deliveries.csv:")
        print("1. Visit: https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set")
        print("2. Download the dataset")
        print("3. Extract 'deliveries.csv' to this folder")
        print("4. Run this script again")

    except Exception as e:
        print(f"\nERROR: An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
