import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
import joblib
import json
import os
import glob
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional

class SequentialPumpTrainer:
    def __init__(self, 
                 pump_threshold: float = 55000,  # Market cap threshold for pump
                 window_size: int = 10,          # Each window is 10 seconds
                 confidence_threshold: float = 0.7):
        self.pump_threshold = pump_threshold
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # Initialize model with conservative parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = None

    def get_sequence_features(self, windows: List[Dict]) -> Dict[str, float]:
        """Extract features from a sequence of windows"""
        if not windows:
            return {}
            
        # Get most recent window metrics
        current = windows[-1]['window_metrics']
        
        # Basic current state features
        features = {
            'current_price': current['price_metrics']['price_end'],
            'current_volume': current['sol_metrics']['total_sol_volume'],
            'current_tx_count': current['transaction_metrics']['transfer_count'],
            'current_unique_accounts': current['transaction_metrics']['unique_accounts'],
            'windows_available': len(windows)
        }
        
        # If we have multiple windows, calculate trends
        if len(windows) > 1:
            # Get sequences for key metrics
            prices = [w['window_metrics']['price_metrics']['price_end'] for w in windows]
            volumes = [w['window_metrics']['sol_metrics']['total_sol_volume'] for w in windows]
            txs = [w['window_metrics']['transaction_metrics']['transfer_count'] for w in windows]
            accounts = [w['window_metrics']['transaction_metrics']['unique_accounts'] for w in windows]
            
            # Calculate changes
            def safe_changes(values: List[float]) -> List[float]:
                if len(values) < 2:
                    return []
                return [(values[i] - values[i-1]) / max(values[i-1], 1e-8) 
                        for i in range(1, len(values))]
            
            # Price trends
            price_changes = safe_changes(prices)
            if price_changes:
                features.update({
                    'price_trend': np.mean(price_changes),
                    'price_volatility': np.std(price_changes) if len(price_changes) > 1 else 0,
                    'price_acceleration': np.mean(np.diff(price_changes)) if len(price_changes) > 1 else 0,
                    'total_price_change': (prices[-1] / prices[0] - 1) if prices[0] > 0 else 0
                })
            
            # Volume trends    
            volume_changes = safe_changes(volumes)
            if volume_changes:
                features.update({
                    'volume_trend': np.mean(volume_changes),
                    'volume_volatility': np.std(volume_changes) if len(volume_changes) > 1 else 0,
                    'volume_acceleration': np.mean(np.diff(volume_changes)) if len(volume_changes) > 1 else 0,
                    'total_volume_change': (volumes[-1] / max(volumes[0], 1e-8) - 1)
                })
            
            # Transaction trends
            tx_changes = safe_changes(txs)
            if tx_changes:
                features.update({
                    'tx_trend': np.mean(tx_changes),
                    'tx_volatility': np.std(tx_changes) if len(tx_changes) > 1 else 0,
                    'tx_acceleration': np.mean(np.diff(tx_changes)) if len(tx_changes) > 1 else 0,
                    'total_tx_change': (txs[-1] / max(txs[0], 1) - 1)
                })
            
            # Account growth
            account_changes = safe_changes(accounts)
            if account_changes:
                features.update({
                    'account_growth': np.mean(account_changes),
                    'total_account_change': (accounts[-1] / max(accounts[0], 1) - 1)
                })
        
        return features

    def will_pump_later(self, current_timestamp: int, price_data: List[Dict]) -> bool:
        """Check if token pumps above threshold after current timestamp"""
        future_prices = [
            p['market_cap'] for p in price_data 
            if p['timestamp'] > current_timestamp
        ]
        return any(price >= self.pump_threshold for price in future_prices)

    def prepare_sequence_data(self, 
                            enhanced_data: List[Dict], 
                            price_data: List[Dict]) -> List[Tuple[Dict, bool]]:
        """Prepare training data from progressive window sequences"""
        
        training_data = []
        
        # Sort data chronologically
        enhanced_data = sorted(enhanced_data, key=lambda x: x['timestamp'])
        price_data = sorted(price_data, key=lambda x: x['timestamp'])
        
        # Process each possible sequence length
        current_sequence = []
        
        for window in enhanced_data:
            # Add window to current sequence
            current_sequence.append(window)
            
            # Extract features from available windows
            features = self.get_sequence_features(current_sequence)
            
            # Check if pump happens after current window
            current_timestamp = window['timestamp']
            will_pump = self.will_pump_later(current_timestamp, price_data)
            
            # Store example with sequence length
            training_data.append({
                'features': features,
                'will_pump': will_pump,
                'sequence_length': len(current_sequence)
            })
            
        return training_data

    def train(self):
        """Train the sequential pump predictor"""
        print("\nStarting sequential pump prediction training...")
        
        all_features = []
        all_labels = []
        sequence_lengths = []
        
        # Find and process all data files
        enhanced_files = glob.glob("yes/enhanced_*.json")
        files_processed = 0
        
        for enhanced_file in enhanced_files:
            try:
                # Get matching price file
                token_id = os.path.basename(enhanced_file).split('_')[1]
                price_files = glob.glob(f"yes/price_history_{token_id}_*.json")
                
                if not price_files:
                    continue
                    
                print(f"\nProcessing {token_id}")
                
                # Load data
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
                with open(price_files[0], 'r') as f:
                    price_data = json.load(f)
                
                # Get training sequences
                sequences = self.prepare_sequence_data(enhanced_data, price_data)
                
                for seq in sequences:
                    if seq['features']:  # Only add if features were extracted
                        all_features.append(seq['features'])
                        all_labels.append(seq['will_pump'])
                        sequence_lengths.append(seq['sequence_length'])
                        
                files_processed += 1
                
                # Print sequence statistics
                pump_seqs = sum(1 for s in sequences if s['will_pump'])
                print(f"Added {len(sequences)} sequences ({pump_seqs} future pumps)")
                
            except Exception as e:
                print(f"Error processing {enhanced_file}: {str(e)}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found!")
            
        # Convert to DataFrame
        X = pd.DataFrame(all_features)
        y = np.array(all_labels)
        
        self.feature_columns = X.columns
        
        # Print statistics by sequence length
        print("\nPrediction statistics by sequence length:")
        df = pd.DataFrame({
            'sequence_length': sequence_lengths,
            'will_pump': y
        })
        stats = df.groupby('sequence_length')['will_pump'].agg(['count', 'mean'])
        print(stats)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nOverall Performance:")
        print(f"Training accuracy: {train_score:.2f}")
        print(f"Test accuracy: {test_score:.2f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Evaluate performance by sequence length
        test_df = pd.DataFrame({
            'sequence_length': sequence_lengths[-len(y_test):],
            'true_label': y_test,
            'predicted': predictions,
            'confidence': probabilities
        })
        
        print("\nPerformance by sequence length:")
        for length in sorted(test_df['sequence_length'].unique()):
            mask = test_df['sequence_length'] == length
            if sum(mask) > 0:
                length_preds = test_df[mask]
                accuracy = (length_preds['true_label'] == length_preds['predicted']).mean()
                avg_conf = length_preds['confidence'].mean()
                print(f"Length {length:2d}: Accuracy = {accuracy:.2f}, Avg Confidence = {avg_conf:.2f}")
        
        # Feature importance
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        print("\nTop 10 Important Features:")
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{feat}: {imp:.4f}")
            
        return test_score

    def save_model(self, filepath: str = "sequential_pump_detector.joblib"):
        """Save the trained model and configuration"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'pump_threshold': self.pump_threshold,
            'window_size': self.window_size,
            'confidence_threshold': self.confidence_threshold,
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")

def train_model():
    trainer = SequentialPumpTrainer()
    accuracy = trainer.train()
    trainer.save_model()
    return accuracy

if __name__ == "__main__":
    train_model()