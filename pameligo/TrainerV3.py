import os
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from typing import List, Dict, Tuple
from datetime import datetime
import re
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.base import clone

class WindowTracker:
    def __init__(self, save_path="training_windows_analysis.json"):
        self.save_path = save_path
        self.windows = []
        
    def track_window(self, 
                window_start: int,
                current_mcap: float,
                target_mcap: float,
                future_prices: list,
                features: dict,
                label: int,
                prediction: float = None) -> None:
        """Track a training window and its outcome"""
        # Convert numpy values to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return [convert_to_native(x) for x in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            return obj

        # Convert features to native Python types
        features = convert_to_native(features)
        future_prices = convert_to_native(future_prices)

        window = {
            'timestamp': int(window_start),
            'datetime': datetime.fromtimestamp(window_start).strftime('%Y-%m-%d %H:%M:%S'),
            'current_mcap': float(current_mcap),
            'target_mcap': float(target_mcap),
            'mcap_needed': float(target_mcap - current_mcap),
            'future_prices_available': len(future_prices),
            'future_prices': {
                'count': len(future_prices),
                'max': max(future_prices) if future_prices else None,
                'timestamps': [int(window_start + (i+1)*10) for i in range(len(future_prices))],
                'values': future_prices
            },
            'assigned_label': int(label),
            'model_prediction': float(prediction) if prediction is not None else None,
            'features': features,
            'achieved_pump': bool(label),
            'mcap_increase': float(max(future_prices) - current_mcap) if future_prices else None
        }
        
        self.windows.append(window)
        self.save_analysis()
        
    def save_analysis(self):
        """Save window analysis to JSON file"""
        with open(self.save_path, 'w') as f:
            json.dump({
                'total_windows': len(self.windows),
                'windows': self.windows,
                'summary': self.get_summary()
            }, f, indent=2)
            
    def get_summary(self) -> dict:
        """Generate summary statistics"""
        window_count = len(self.windows)
        pump_count = sum(1 for w in self.windows if w['achieved_pump'])
        
        # Analyze future price windows
        future_lengths = [w['future_prices']['count'] for w in self.windows]
        avg_future_window = sum(future_lengths) / len(future_lengths) if future_lengths else 0
        
        return {
            'training_data': {
                'total_windows': window_count,
                'pump_windows': pump_count,
                'no_pump_windows': window_count - pump_count,
                'pump_percentage': (pump_count / window_count * 100) if window_count > 0 else 0
            },
            'future_visibility': {
                'avg_future_prices': avg_future_window,
                'min_future_prices': min(future_lengths) if future_lengths else 0,
                'max_future_prices': max(future_lengths) if future_lengths else 0
            },
            'verification': {
                'windows_with_incomplete_data': sum(1 for w in self.windows if w['future_prices']['count'] < 6),
                'suspiciously_high_success': sum(1 for w in self.windows if w['achieved_pump'] and w['mcap_increase'] > 50000)
            }
        }

class PricePredictionTrainer:
    def __init__(self, target_mcap_increase: float = 10000, window_size: int = 10, max_gap: int = 40):
        self.target_mcap_increase = target_mcap_increase
        self.window_size = window_size 
        self.max_gap = max_gap
        self.model = None
        self.window_tracker = WindowTracker()
        
    def extract_max_mcap(self, filename: str) -> float:
        """Extract max market cap from filename for validation."""
        match = re.search(r'_(\d+)\.json$', filename)
        return float(match.group(1)) if match else 0
        
    def load_and_prepare_data(self, price_file: str, enhanced_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Load and prepare data from price and enhanced files."""
        with open(price_file, 'r') as f:
            price_data = pd.DataFrame(json.load(f))
        with open(enhanced_file, 'r') as f:
            enhanced_data = pd.DataFrame(json.load(f))
            
        # Sort by timestamp
        price_data = price_data.sort_values('timestamp')
        enhanced_data = enhanced_data.sort_values('timestamp')
        
        # Get max_mcap for validation
        max_mcap = self.extract_max_mcap(price_file)
        
        return price_data, enhanced_data, max_mcap
        
    def create_windows(self, price_data: pd.DataFrame, enhanced_data: pd.DataFrame) -> List[Dict]:
        """Create time windows from the data."""
        windows = []
        start_time = price_data['timestamp'].min()
        end_time = price_data['timestamp'].max()
        
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + self.window_size
            
            # Get window data
            window_prices = price_data[
                (price_data['timestamp'] >= current_time) & 
                (price_data['timestamp'] < window_end)
            ]
            
            window_enhanced = enhanced_data[
                (enhanced_data['timestamp'] >= current_time) & 
                (enhanced_data['timestamp'] < window_end)
            ]
            
            # Skip if no transactions or gap too large
            if len(window_prices) == 0 or \
               (len(windows) > 0 and current_time - windows[-1]['end_time'] > self.max_gap):
                current_time += self.window_size
                continue
                
            # Create window dict
            window = {
                'start_time': current_time,
                'end_time': window_end,
                'price_data': window_prices,
                'enhanced_data': window_enhanced,
                'last_price': window_prices.iloc[-1]['price_in_usd']
            }
            
            windows.append(window)
            current_time += self.window_size
            
        return windows
        
    def create_label(self, window: Dict, full_price_data: pd.DataFrame) -> Tuple[int, float]:
        """Create label with tracking"""
        current_mcap = window['price_data'].iloc[-1]['market_cap']
        target_mcap = current_mcap + self.target_mcap_increase
        
        # Get future market caps
        future_mcaps = full_price_data[
            full_price_data['timestamp'] > window['end_time']
        ]['market_cap'].tolist()
        
        # Extract features before tracking
        features = self.extract_features(window)
        
        if len(future_mcaps) == 0:
            return 0, 0
            
        max_future_mcap = max(future_mcaps)
        achieved_increase = max_future_mcap - current_mcap
        
        is_positive = achieved_increase >= self.target_mcap_increase
        
        return int(is_positive), achieved_increase

    def extract_features(self, window: Dict) -> Dict:
        """Extract features from a time window."""
        price_data = window['price_data']
        enhanced_data = window['enhanced_data']
        
        # Market Cap and Price Features
        market_features = {
            'mcap_current': price_data.iloc[-1]['market_cap'],
            'mcap_mean': price_data['market_cap'].mean(),
            'mcap_std': price_data['market_cap'].std(),
            'mcap_min': price_data['market_cap'].min(),
            'mcap_max': price_data['market_cap'].max(),
            'mcap_change': price_data['market_cap'].pct_change().mean(),
            'mcap_acceleration': price_data['market_cap'].diff().diff().mean(),
            'price_change': price_data['price_in_usd'].pct_change().mean(),
            'price_volatility': price_data['price_in_usd'].std() / price_data['price_in_usd'].mean()
        }
        
        # Volume Features
        volume_features = {
            'volume_mean': price_data['sol_amount'].mean(),
            'volume_total': price_data['sol_amount'].sum(),
            'tx_count': len(price_data),
            'avg_tx_size': price_data['sol_amount'].mean(),
            'volume_acceleration': price_data['sol_amount'].diff().diff().mean()
        }
        
        # Enhanced metrics if available
        if len(enhanced_data) > 0:
            last_metrics = enhanced_data.iloc[-1]['window_metrics']
            enhanced_features = {
                'tx_density': last_metrics['transaction_metrics']['tx_density'],
                'unique_accounts': last_metrics['transaction_metrics']['unique_accounts'],
                'sol_flow_volatility': last_metrics['sol_metrics']['sol_flow_volatility'],
                'interaction_concentration': last_metrics['account_metrics']['interaction_concentration'],
                'total_fees': last_metrics['fee_metrics']['total_fees'],
                'program_calls': last_metrics['program_metrics']['total_program_calls']
            }
        else:
            enhanced_features = {
                'tx_density': 0,
                'unique_accounts': 0,
                'sol_flow_volatility': 0,
                'interaction_concentration': 0,
                'total_fees': 0,
                'program_calls': 0
            }
            
        return {**market_features, **volume_features, **enhanced_features}
        
    def prepare_training_data(self, windows: List[Dict], price_data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data with proper prediction tracking"""
        features = []
        labels = []
        window_info = []  # Store window information for tracking
        
        # First pass: collect features and labels
        for window in windows:
            window_features = self.extract_features(window)
            label, achieved_increase = self.create_label(window, price_data)
            
            features.append(window_features)
            labels.append(label)
            
            # Store window information
            window_info.append({
                'window': window,
                'features': window_features,
                'label': label,
                'future_prices': price_data[price_data['timestamp'] > window['end_time']]['market_cap'].tolist()
            })

        X = pd.DataFrame(features)
        y = np.array(labels)

        # Generate predictions using cross-validation
        predictions = np.zeros(len(y))
        if len(X) >= 5:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y[train_idx]
                
                # Train fold model
                fold_model = LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    is_unbalanced=True,
                    num_leaves=31,
                    learning_rate=0.05,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    verbose=-1
                )
                fold_model.fit(X_train, y_train)
                
                # Generate predictions for validation fold
                predictions[val_idx] = fold_model.predict_proba(X_val)[:, 1]
        else:
            # For small datasets, use simple predictions
            temp_model = LGBMClassifier(
                objective='binary',
                metric='auc',
                is_unbalanced=True
            )
            temp_model.fit(X, y)
            predictions = temp_model.predict_proba(X)[:, 1]

        # Track windows with their predictions
        for idx, info in enumerate(window_info):
            self.window_tracker.track_window(
                window_start=info['window']['start_time'],
                current_mcap=info['window']['price_data'].iloc[-1]['market_cap'],
                target_mcap=info['window']['price_data'].iloc[-1]['market_cap'] + self.target_mcap_increase,
                future_prices=info['future_prices'],
                features=info['features'],
                label=info['label'],
                prediction=float(predictions[idx])  # Convert to float to ensure JSON serialization
            )

        return X, y

    def evaluate_model(self, X, y):
        """Evaluate the trained model and return predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)

        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y, y_pred))

        return y_pred, y_pred_proba

    def train(self, pred_yes_folder: str = 'PredYes'):
        """Train the model using data from the PredYes folder."""
        all_features = []
        all_labels = []
        
        print(f"Starting training with target market cap increase: ${self.target_mcap_increase}")
        
        # Process all files in folder
        for filename in os.listdir(pred_yes_folder):
            if not filename.startswith('price_history_'):
                continue
                
            # Get matching enhanced file
            token_id = filename.split('_')[2]
            timestamp = '_'.join(filename.split('_')[3:]).split('.')[0]
            enhanced_filename = f'enhanced_{token_id}_{timestamp}.json'
            price_file = os.path.join(pred_yes_folder, filename)
            enhanced_file = os.path.join(pred_yes_folder, enhanced_filename)
            
            if not os.path.exists(enhanced_file):
                continue
                
            print(f"\nProcessing {filename}...")
            max_mcap = self.extract_max_mcap(filename)
            print(f"Max market cap from filename: ${max_mcap}")
            
            # Load and prepare data
            price_data, enhanced_data, _ = self.load_and_prepare_data(
                price_file, enhanced_file)
            
            print(f"Market Cap range: ${price_data['market_cap'].min():.2f} to ${price_data['market_cap'].max():.2f}")
            
            # Create windows
            windows = self.create_windows(price_data, enhanced_data)
            print(f"Created {len(windows)} windows")
            
            # Prepare training data
            X, y = self.prepare_training_data(windows, price_data)
            print(f"Positive cases in this file: {sum(y)}")
            
            if len(X) > 0:
                all_features.append(X)
                all_labels.append(y)

        # Combine all data
        if not all_features:
            raise ValueError("No valid training data found!")
            
        X = pd.concat(all_features, axis=0)
        y = np.concatenate(all_labels)
        
        # Train model
        print("\nTraining final model...")
        print(f"Total samples: {len(X)}")
        print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
        
        self.model = LGBMClassifier(
            objective='binary',
            metric='auc',
            is_unbalanced=True,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1
        )
        
        self.model.fit(X, y)
        
        # Generate and store final predictions
        final_predictions = self.model.predict_proba(X)[:, 1]
        
        # Update all window predictions with final model predictions
        for idx, window in enumerate(self.window_tracker.windows):
            if window['model_prediction'] is None:
                window['model_prediction'] = float(final_predictions[idx])
        
        # Print feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # Evaluate model
        print("\nEvaluating model on training data...")
        self.predictions, self.prediction_probabilities = self.evaluate_model(X, y)
        
        # Print prediction distribution
        probs = self.prediction_probabilities
        print("\nPrediction Distribution:")
        print(f"  0-25%: {sum(probs < 0.25)} predictions")
        print(f"  25-50%: {sum((probs >= 0.25) & (probs < 0.50))} predictions")
        print(f"  50-75%: {sum((probs >= 0.50) & (probs < 0.75))} predictions")
        print(f"  75-100%: {sum(probs >= 0.75)} predictions")
        
        # Save final window analysis
        self.window_tracker.save_analysis()
        
        return self.model

        
    def save_model(self, filepath: str = "mcap_prediction_model.joblib"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        model_data = {
            'model': self.model,
            'target_mcap_increase': self.target_mcap_increase,
            'window_size': self.window_size,
            'max_gap': self.max_gap,
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")

def train_model(pred_yes_folder: str = 'newyesV2', 
                target_mcap_increase: float = 10000,
                save_path: str = 'mcap_prediction_model.joblib'):
    """Convenience function to train, evaluate, and save model."""
    trainer = PricePredictionTrainer(target_mcap_increase=target_mcap_increase)
    trainer.train(pred_yes_folder)
    trainer.save_model(save_path)

if __name__ == "__main__":
    train_model()