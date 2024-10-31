# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.

import pandas as pd
import numpy as np
from pycaret.classification import *
import json
import os
from pathlib import Path



def load_transaction_file(file_path):
    """
    Load a single JSON transaction file and process it
    """
    try:
        with open(file_path, 'r') as f:
            transactions = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Sort by timestamp to ensure order
        df = df.sort_values('timestamp')
        
        # Add features specific to pump detection
        df['price_change'] = df['price_in_usd'].pct_change()
        df['mcap_change'] = df['market_cap'].pct_change()
        df['volume'] = df['token_amount'] * df['price_in_usd']
        
        # Calculate rolling metrics (5-point windows)
        df['price_volatility'] = df['price_change'].rolling(5).std()
        df['mcap_volatility'] = df['mcap_change'].rolling(5).std()
        df['price_acceleration'] = df['price_change'].diff()
        df['volume_change'] = df['volume'].pct_change()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def load_and_prepare_data_from_folders(yes_folder='yes', no_folder='no', test_folder='test'):
    """
    Load and process transaction data from all folders
    """
    train_dfs = []
    test_dfs = []
    
    # Load pump patterns (yes folder)
    if os.path.exists(yes_folder):
        print("\nLoading pump patterns...")
        for file in Path(yes_folder).glob('*.json'):
            df = load_transaction_file(file)
            if not df.empty:
                df['is_pump'] = 1
                df['data_source'] = 'train'
                train_dfs.append(df)
                print(f"Loaded pump pattern: {file.name} ({len(df)} records)")
    
    # Load non-pump patterns (no folder)
    if os.path.exists(no_folder):
        print("\nLoading non-pump patterns...")
        for file in Path(no_folder).glob('*.json'):
            df = load_transaction_file(file)
            if not df.empty:
                df['is_pump'] = 0
                df['data_source'] = 'train'
                train_dfs.append(df)
                print(f"Loaded non-pump pattern: {file.name} ({len(df)} records)")
    
    # Load test patterns
    if os.path.exists(test_folder):
        print("\nLoading test patterns...")
        for file in Path(test_folder).glob('*.json'):
            df = load_transaction_file(file)
            if not df.empty:
                df['data_source'] = 'test'
                test_dfs.append(df)
                print(f"Loaded test pattern: {file.name} ({len(df)} records)")
    
    # Combine all data
    train_data = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_data = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    # Print summary
    print("\nData Loading Summary:")
    if not train_data.empty:
        print(f"Training samples (pump): {len(train_data[train_data['is_pump'] == 1])}")
        print(f"Training samples (non-pump): {len(train_data[train_data['is_pump'] == 0])}")
    if not test_data.empty:
        print(f"Test samples: {len(test_data)}")
    
    return train_data, test_data

def train_pump_detector():
    """
    Train the pump detector
    """
    # Load data
    train_df, test_df = load_and_prepare_data_from_folders()
    
    if train_df.empty:
        raise ValueError("No training data found! Please check yes/no folders.")
    
    # Initialize PyCaret
    print("\nInitializing PyCaret...")
    clf = setup(
        data=train_df,
        target='is_pump',
        session_id=123,
        numeric_features=[
            'price_in_usd',
            'market_cap',
            'price_change',
            'mcap_change',
            'price_volatility',
            'mcap_volatility',
            'price_acceleration',
            'volume',
            'volume_change'
        ],
        ignore_features=['timestamp', 'date', 'signature', 'data_source', 
                        'token_amount', 'sol_amount', 'price_in_sol'],
        normalize=True,
        feature_selection=True,
        remove_multicollinearity=True
    )
    
    # Train model
    print("\nTraining models...")
    best_model = compare_models(n_select=1)
    
    # Tune the model
    print("\nFine-tuning model...")
    tuned_model = tune_model(best_model)
    
    # Save model
    final_model = finalize_model(tuned_model)
    save_model(final_model, 'pump_detector_final')
    
    # Make predictions on test data
    if not test_df.empty:
        print("\nMaking predictions on test data...")
        test_predictions = predict_model(final_model, data=test_df)
        test_predictions.to_csv('test_predictions.csv', index=False)
        print("Test predictions saved to 'test_predictions.csv'")
    
    return final_model

if __name__ == "__main__":
    try:
        model = train_pump_detector()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")