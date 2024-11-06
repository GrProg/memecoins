import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
import glob
from datetime import datetime

class DumpTrainer:
    def __init__(self, mcap_threshold=55000):
        self.mcap_threshold = mcap_threshold
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
        self.feature_columns = None
        
    def extract_features(self, enhanced_data, price_data):
        """Extract features with enhanced time-based metrics"""
        # Sort both datasets by timestamp ascending
        sorted_enhanced = sorted(enhanced_data, key=lambda x: x['timestamp'])
        sorted_prices = sorted(price_data, key=lambda x: x['timestamp'])
        
        features = []
        
        # Calculate time-based metrics over full dataset
        total_time_span = max(1, sorted_prices[-1]['timestamp'] - sorted_prices[0]['timestamp'])
        initial_mcap = sorted_prices[0]['market_cap']
        current_mcap = sorted_prices[-1]['market_cap']
        mcap_growth_rate = (current_mcap - initial_mcap) / total_time_span
        
        for window in sorted_enhanced:
            window_time = window['timestamp']
            metrics = window['window_metrics']
            
            # Get price data for this window
            window_prices = [p for p in sorted_prices if abs(p['timestamp'] - window_time) <= 20]
            if len(window_prices) < 2:
                continue
                
            # Calculate time-based features
            window_features = {
                # Existing metrics
                'price_volatility': metrics['price_metrics']['price_volatility'],
                'mcap_volatility': metrics['price_metrics']['mcap_volatility'],
                'price_change': metrics['price_metrics']['price_change'],
                'tx_density': metrics['transaction_metrics']['tx_density'],
                'transfer_count': metrics['transaction_metrics']['transfer_count'],
                'unique_accounts': metrics['transaction_metrics']['unique_accounts'],
                'total_sol_volume': metrics['sol_metrics']['total_sol_volume'],
                'sol_flow_volatility': metrics['sol_metrics']['sol_flow_volatility'],
                'avg_sol_per_tx': metrics['sol_metrics']['avg_sol_per_tx'],
                'interaction_concentration': metrics['account_metrics']['interaction_concentration'],
                'avg_interactions': metrics['account_metrics']['avg_interactions_per_account'],
                'unique_programs': metrics['program_metrics']['unique_programs'],
                'avg_calls_per_program': metrics['program_metrics']['avg_calls_per_program'],
                
                # New time-based metrics with safety checks
                'mcap_growth_rate': mcap_growth_rate,
                'time_to_threshold': (55000 - current_mcap) / max(1e-6, mcap_growth_rate) if mcap_growth_rate > 0 else 1e6,
                'acceleration': (window_prices[-1]['market_cap'] - window_prices[0]['market_cap']) / max(1, window_prices[-1]['timestamp'] - window_prices[0]['timestamp']),
                'momentum_score': mcap_growth_rate * metrics['sol_metrics']['total_sol_volume'],
                'buy_pressure': (metrics['sol_metrics']['total_sol_volume'] / max(1, window_prices[-1]['timestamp'] - window_prices[0]['timestamp'])) * (window_prices[-1]['market_cap'] / max(1, window_prices[0]['market_cap'])),
                'volatility_trend': metrics['price_metrics']['price_volatility'] / max(1, window_time - sorted_prices[0]['timestamp']),
                'volume_acceleration': metrics['sol_metrics']['total_sol_volume'] / max(1, window_time - sorted_prices[0]['timestamp']),
                'price_efficiency': abs(window_prices[-1]['market_cap'] - window_prices[0]['market_cap']) / max(1e-6, metrics['sol_metrics']['total_sol_volume']),
                'time_weighted_tx_density': metrics['transaction_metrics']['tx_density'] * (window_time - sorted_prices[0]['timestamp']),
                'relative_progress': (current_mcap - initial_mcap) / max(1, 55000 - initial_mcap)
            }
            
            features.append(window_features)
        
        return pd.DataFrame(features)

    def check_dump_after_threshold(self, price_data):
        """Enhanced dump detection with time consideration"""
        sorted_prices = sorted(price_data, key=lambda x: x['timestamp'])
        reached_threshold = False
        max_mcap = 0
        time_above_threshold = 0

        for i, data in enumerate(sorted_prices):
            mcap = data['market_cap']

            if mcap >= self.mcap_threshold:
                if not reached_threshold:
                    reached_threshold = True
                    threshold_idx = i

                max_mcap = max(max_mcap, mcap)

            elif reached_threshold:
                # Calculate dump metrics
                time_to_dump = sorted_prices[i]['timestamp'] - sorted_prices[threshold_idx]['timestamp']
                drop = (mcap - max_mcap) / max(1, max_mcap)

                if drop <= -0.15:  # 15% drop
                    return True, time_to_dump

        return False, None

    def train(self):
        """Train model using data from yes folder"""
        print("\nStarting training process...")
        print("Loading data from 'yes' folder...")
        
        # Initialize data collectors
        all_features = []
        all_labels = []
        files_processed = 0
        
        # Find all enhanced files
        enhanced_files = glob.glob("yes/enhanced_*.json")
        
        for enhanced_file in enhanced_files:
            try:
                # Get matching price file
                base_name = os.path.basename(enhanced_file)
                token_id = base_name.split('_')[1]
                timestamp = '_'.join(base_name.split('_')[2:]).replace('.json', '')
                price_file = os.path.join("yes", f"price_history_{token_id}_{timestamp}.json")
                
                if not os.path.exists(price_file):
                    print(f"Skipping {token_id} - No matching price file found")
                    continue
                
                print(f"\nProcessing token {token_id}...")
                
                # Load both files
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
                with open(price_file, 'r') as f:
                    price_data = json.load(f)
                
                # Extract features
                features = self.extract_features(enhanced_data, price_data)
                
                if features.empty:
                    print(f"Skipping {token_id} - No valid features extracted")
                    continue
                
                # Get label
                did_dump, _ = self.check_dump_after_threshold(price_data)
                labels = [did_dump] * len(features)
                
                all_features.append(features)
                all_labels.extend(labels)
                files_processed += 1
                
                print(f"✓ Processed {len(features)} windows, dump={did_dump}")
                
            except Exception as e:
                print(f"Error processing {enhanced_file}: {str(e)}")
                continue
            
        if not all_features:
            raise ValueError("No valid training data found!")
                
        # Combine all data
        X = pd.concat(all_features, axis=0)
        y = np.array(all_labels)
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"\nTraining Results:")
        print(f"Files processed: {files_processed}")
        print(f"Total samples: {len(X)}")
        print(f"Test accuracy: {accuracy:.2%}")
        
        # Show feature importance
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        print("\nTop 5 important features:")
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{feat}: {imp:.4f}")
        
        return accuracy

    def save_model(self, filepath="dump_detector.joblib"):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'mcap_threshold': self.mcap_threshold
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")

def train_model():
    trainer = DumpTrainer()
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train_model()