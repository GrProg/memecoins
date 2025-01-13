import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import json
import os
import glob
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional

class PumpTrainer:
    def __init__(self):
        # Core model parameters
        self.model = RandomForestClassifier(
            n_estimators=100,        # Reduced from 200 to prevent overfitting
            max_depth=8,             # Reduced from 12 to prevent overfitting
            min_samples_split=10,    # Increased to require more evidence for splits
            min_samples_leaf=5,      # Increased to prevent overfitting on small patterns
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Pattern detection thresholds
        self.thresholds = {
            'short_window': 3,       # Number of windows for short-term analysis
            'medium_window': 5,      # Number of windows for medium-term analysis
            
            # Price thresholds
            'price_surge': 0.10,     # 10% minimum price increase
            'price_volatility': 0.15, # 15% minimum price volatility
            
            # Volume thresholds
            'min_sol_volume': 0.5,   # Minimum SOL volume to consider
            'volume_surge': 0.20,    # 20% volume increase
            
            # Transaction thresholds
            'min_transactions': 5,    # Minimum transactions to consider
            'tx_surge': 0.25,        # 25% transaction increase
            
            # Account thresholds
            'min_accounts': 3,        # Minimum unique accounts
            'account_growth': 0.15    # 15% unique account growth
        }

    def calculate_window_changes(self, windows: List[Dict], lookback: int = 3) -> Dict[str, float]:
        """Calculate relative changes over specified lookback period"""
        if len(windows) < lookback:
            return {}
            
        current = windows[-1]['window_metrics']
        past = windows[-lookback]['window_metrics']
        
        def safe_change(current_val, past_val):
            if past_val == 0:
                return 0
            return (current_val - past_val) / past_val
            
        # Calculate primary metric changes
        changes = {
            'price_change': safe_change(
                current['price_metrics']['price_end'],
                past['price_metrics']['price_end']
            ),
            'volume_change': safe_change(
                current['sol_metrics']['total_sol_volume'],
                past['sol_metrics']['total_sol_volume']
            ),
            'tx_change': safe_change(
                current['transaction_metrics']['transfer_count'],
                past['transaction_metrics']['transfer_count']
            ),
            'account_change': safe_change(
                current['transaction_metrics']['unique_accounts'],
                past['transaction_metrics']['unique_accounts']
            )
        }
        
        # Calculate volatilities
        price_sequence = [w['window_metrics']['price_metrics']['price_end'] for w in windows[-lookback:]]
        volume_sequence = [w['window_metrics']['sol_metrics']['total_sol_volume'] for w in windows[-lookback:]]
        
        changes.update({
            'price_volatility': np.std(price_sequence) / np.mean(price_sequence) if len(price_sequence) > 1 else 0,
            'volume_volatility': np.std(volume_sequence) / np.mean(volume_sequence) if len(volume_sequence) > 1 else 0
        })
        
        return changes

    def detect_pump_pattern(self, windows: List[Dict]) -> Dict[str, float]:
        """
        Detect pump patterns using multiple timeframes and signals.
        Returns confidence scores for different aspects of the pattern.
        """
        if len(windows) < self.thresholds['medium_window']:
            return {'is_pump': False, 'confidence': 0.0, 'signals': {}}
            
        # Get changes over different timeframes
        short_changes = self.calculate_window_changes(windows, self.thresholds['short_window'])
        medium_changes = self.calculate_window_changes(windows, self.thresholds['medium_window'])
        
        current = windows[-1]['window_metrics']
        
        # Check volume requirements
        sufficient_volume = (
            current['sol_metrics']['total_sol_volume'] >= self.thresholds['min_sol_volume'] and
            current['transaction_metrics']['transfer_count'] >= self.thresholds['min_transactions']
        )
        
        if not sufficient_volume:
            return {'is_pump': False, 'confidence': 0.0, 'signals': {}}
        
        # Calculate individual signal strengths
        signals = {
            'price_surge': max(0, min(1, short_changes.get('price_change', 0) / self.thresholds['price_surge'])),
            'volume_surge': max(0, min(1, short_changes.get('volume_change', 0) / self.thresholds['volume_surge'])),
            'tx_surge': max(0, min(1, short_changes.get('tx_change', 0) / self.thresholds['tx_surge'])),
            'account_growth': max(0, min(1, short_changes.get('account_change', 0) / self.thresholds['account_growth'])),
            'price_volatility': max(0, min(1, short_changes.get('price_volatility', 0) / self.thresholds['price_volatility']))
        }
        
        # Calculate confirmation signals from medium timeframe
        if medium_changes:
            signals.update({
                'sustained_volume': max(0, min(1, medium_changes.get('volume_change', 0) / (self.thresholds['volume_surge'] * 0.5))),
                'sustained_price': max(0, min(1, medium_changes.get('price_change', 0) / (self.thresholds['price_surge'] * 0.5)))
            })
        
        # Calculate overall pump confidence
        required_signals = ['price_surge', 'volume_surge', 'tx_surge']
        supporting_signals = ['account_growth', 'price_volatility', 'sustained_volume', 'sustained_price']
        
        # Primary signals must be strong
        primary_score = np.mean([signals[s] for s in required_signals])
        
        # Supporting signals provide additional confidence
        support_score = np.mean([signals.get(s, 0) for s in supporting_signals])
        
        # Combined confidence score
        confidence = primary_score * 0.7 + support_score * 0.3
        
        # Determine if this is a pump pattern
        is_pump = (
            confidence >= 0.6 and  # Overall confidence threshold
            all(signals[s] >= 0.5 for s in required_signals)  # All required signals must be moderately strong
        )
        
        return {
            'is_pump': is_pump,
            'confidence': confidence,
            'signals': signals
        }

    def prepare_features(self, windows: List[Dict]) -> Dict[str, float]:
        """Extract features from window sequence"""
        if not windows:
            return {}
            
        current = windows[-1]['window_metrics']
        
        # Basic current window features
        features = {
            'price': current['price_metrics']['price_end'],
            'volume': current['sol_metrics']['total_sol_volume'],
            'tx_count': current['transaction_metrics']['transfer_count'],
            'unique_accounts': current['transaction_metrics']['unique_accounts'],
            'avg_tx_size': current['sol_metrics']['avg_sol_per_tx'],
            'concentration': current['account_metrics']['interaction_concentration']
        }
        
        # Add short and medium term changes if available
        if len(windows) >= self.thresholds['short_window']:
            short_changes = self.calculate_window_changes(windows, self.thresholds['short_window'])
            features.update({
                'short_price_change': short_changes.get('price_change', 0),
                'short_volume_change': short_changes.get('volume_change', 0),
                'short_tx_change': short_changes.get('tx_change', 0),
                'short_price_volatility': short_changes.get('price_volatility', 0),
                'short_volume_volatility': short_changes.get('volume_volatility', 0)
            })
            
        if len(windows) >= self.thresholds['medium_window']:
            medium_changes = self.calculate_window_changes(windows, self.thresholds['medium_window'])
            features.update({
                'medium_price_change': medium_changes.get('price_change', 0),
                'medium_volume_change': medium_changes.get('volume_change', 0),
                'medium_tx_change': medium_changes.get('tx_change', 0),
                'medium_price_volatility': medium_changes.get('price_volatility', 0),
                'medium_volume_volatility': medium_changes.get('volume_volatility', 0)
            })
            
        return features

    def prepare_training_data(self, enhanced_data: List[Dict]) -> List[Tuple[Dict, bool]]:
        """Prepare training examples from token data"""
        training_data = []
        windows = []
        
        # Process windows sequentially
        for window in sorted(enhanced_data, key=lambda x: x['timestamp']):
            windows.append(window)
            
            if len(windows) >= self.thresholds['medium_window']:
                # Extract features
                features = self.prepare_features(windows)
                
                # Detect pump pattern
                pattern = self.detect_pump_pattern(windows)
                
                training_data.append((features, pattern['is_pump'], pattern['confidence']))
                
        return training_data

    def train(self):
        """Train the model using sequential window data"""
        print("\nStarting pump pattern training...")
        
        all_features = []
        all_labels = []
        confidences = []
        
        # Process all enhanced data files
        enhanced_files = glob.glob("yes/enhanced_*.json")
        files_processed = 0
        
        print(f"Found {len(enhanced_files)} token datasets")
        
        for enhanced_file in enhanced_files:
            try:
                print(f"\nProcessing {os.path.basename(enhanced_file)}")
                
                # Load and prepare data
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
                    
                training_examples = self.prepare_training_data(enhanced_data)
                
                if training_examples:
                    features, labels, confs = zip(*training_examples)
                    all_features.extend(features)
                    all_labels.extend(labels)
                    confidences.extend(confs)
                    
                    files_processed += 1
                    print(f"Added {len(training_examples)} examples "
                          f"({sum(labels)} pumps, {len(labels)-sum(labels)} non-pumps)")
                
            except Exception as e:
                print(f"Error processing {enhanced_file}: {str(e)}")
                continue
                
        if not all_features:
            raise ValueError("No valid training data found!")
            
        # Prepare training data
        X = pd.DataFrame(all_features)
        y = np.array(all_labels)
        
        print(f"\nClass distribution:")
        print(f"Pump patterns: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"Non-pump patterns: {len(y)-sum(y)} ({(1-sum(y)/len(y))*100:.1f}%)")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        self.feature_columns = X.columns
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        predictions = self.model.predict(X_test_scaled)
        
        print("\nModel Performance:")
        print(f"Training accuracy: {train_score:.2f}")
        print(f"Test accuracy: {test_score:.2f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Feature importance
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        print("\nTop 10 Important Features:")
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{feat}: {imp:.4f}")
            
        return test_score

    def save_model(self, filepath: str = "pump_detector_v3.joblib"):
        """Save the trained model and configuration"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'thresholds': self.thresholds,
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")

def train_model():
    trainer = PumpTrainer()
    accuracy = trainer.train()
    trainer.save_model()
    return accuracy

if __name__ == "__main__":
    train_model()