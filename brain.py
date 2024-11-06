import pandas as pd
import numpy as np
from pycaret.classification import *
from typing import Dict, List, Any
import json
import os
from pathlib import Path
import re

class ImprovedPumpDataProcessor:
    def __init__(self, baseline_window: int = 5):
        self.baseline_window = baseline_window
        
    def extract_features(self, price_file: str, enhanced_file: str) -> Dict:
        """Extract normalized features focusing on patterns and relative changes"""
        with open(price_file, 'r') as f:
            price_data = pd.DataFrame(json.load(f))
        with open(enhanced_file, 'r') as f:
            enhanced_data = pd.DataFrame(json.load(f))
            
        # Calculate relative price features
        price_features = self._calculate_price_patterns(price_data)
        
        # Calculate volume and transaction patterns
        volume_features = self._calculate_volume_patterns(enhanced_data)
        
        # Calculate account interaction patterns
        account_features = self._calculate_account_patterns(enhanced_data)
        
        # Calculate momentum and acceleration
        momentum_features = self._calculate_momentum_features(price_data, enhanced_data)
        
        return {**price_features, **volume_features, **account_features, **momentum_features}
    
    def _calculate_price_patterns(self, price_data: pd.DataFrame) -> Dict:
        """Calculate price-related patterns with value handling"""
        prices = price_data['price_in_usd'].values
        initial_price = prices[0]
        max_price = max(prices)
        
        # Safe division function
        def safe_div(a, b, default=0):
            try:
                if b == 0:
                    return default
                result = a / b
                return result if np.isfinite(result) else default
            except:
                return default
                
        # Calculate returns safely
        returns = []
        for i in range(1, len(prices)):
            ret = safe_div(prices[i] - prices[i-1], prices[i-1], 0)
            # Cap extreme values
            ret = max(min(ret, 10), -10)  # Cap at 1000% increase/decrease
            returns.append(ret)
        
        returns = np.array(returns)
        
        # Calculate volatility windows safely
        volatility_windows = []
        for i in range(0, len(returns), self.baseline_window):
            window = returns[i:i+self.baseline_window]
            if len(window) >= 3:
                vol = np.std(window)
                if np.isfinite(vol):
                    volatility_windows.append(vol)
        
        return {
            'price_increase_pct': min(safe_div(max_price - initial_price, initial_price) * 100, 1000),
            'time_to_peak_ratio': safe_div(np.argmax(prices), len(prices)),
            'price_volatility_ratio': np.mean(volatility_windows) if volatility_windows else 0,
            'price_acceleration': safe_div(np.mean(np.diff(returns)) if len(returns) > 1 else 0, 1),
            'sustained_growth_ratio': safe_div(np.sum(returns > 0), len(returns)) if len(returns) > 0 else 0
        }
    
    def _calculate_volume_patterns(self, enhanced_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns with value handling"""
        volumes = []
        densities = []
        
        for _, row in enhanced_data.iterrows():
            metrics = row['window_metrics']
            vol = metrics['sol_metrics']['total_sol_volume']
            if np.isfinite(vol):
                volumes.append(vol)
            densities.append(metrics['transaction_metrics']['tx_density'])
        
        def safe_div(a, b, default=0):
            try:
                if b == 0:
                    return default
                result = a / b
                return result if np.isfinite(result) else default
            except:
                return default
        
        baseline_volume = np.mean(volumes[:self.baseline_window]) if len(volumes) >= self.baseline_window else np.mean(volumes)
        max_volume = max(volumes) if volumes else 0
        
        return {
            'volume_spike_ratio': min(safe_div(max_volume, baseline_volume), 1000),
            'volume_acceleration': safe_div(np.mean(np.diff(volumes)) if len(volumes) > 1 else 0, 1),
            'tx_density_spike_ratio': min(safe_div(max(densities), np.mean(densities)), 100),
            'volume_concentration': safe_div(np.std(volumes), np.mean(volumes)),
            'sustained_volume_ratio': safe_div(np.sum(np.array(volumes) > baseline_volume), len(volumes))
        }
    
    def _calculate_account_patterns(self, enhanced_data: pd.DataFrame) -> Dict:
        """Analyze account patterns with value handling"""
        def safe_div(a, b, default=0):
            try:
                if b == 0:
                    return default
                result = a / b
                return result if np.isfinite(result) else default
            except:
                return default
                
        account_metrics = []
        concentrations = []
        
        for _, row in enhanced_data.iterrows():
            metrics = row['window_metrics']['account_metrics']
            account_metrics.append({
                'total_accounts': metrics['total_accounts'],
                'avg_interactions': min(metrics['avg_interactions_per_account'], 100),
                'concentration': min(metrics['interaction_concentration'], 100)
            })
            concentrations.append(min(metrics['interaction_concentration'], 100))
        
        baseline_accounts = np.mean([m['total_accounts'] for m in account_metrics[:self.baseline_window]]) \
                          if len(account_metrics) >= self.baseline_window \
                          else np.mean([m['total_accounts'] for m in account_metrics])
        
        return {
            'account_growth_ratio': min(safe_div(max([m['total_accounts'] for m in account_metrics]), baseline_accounts), 100),
            'interaction_intensity': min(np.mean([m['avg_interactions'] for m in account_metrics]), 100),
            'concentration_spike_ratio': min(safe_div(max(concentrations), np.mean(concentrations)), 100),
            'sustained_activity_ratio': safe_div(
                np.sum(np.array([m['total_accounts'] for m in account_metrics]) > baseline_accounts),
                len(account_metrics)
            )
        }
    
    def _calculate_momentum_features(self, price_data: pd.DataFrame, enhanced_data: pd.DataFrame) -> Dict:
        """Calculate momentum features with value handling"""
        def safe_div(a, b, default=0):
            try:
                if b == 0:
                    return default
                result = a / b
                return result if np.isfinite(result) else default
            except:
                return default
                
        prices = price_data['price_in_usd'].values
        volumes = []
        
        for _, row in enhanced_data.iterrows():
            volumes.append(row['window_metrics']['sol_metrics']['total_sol_volume'])
        
        # Calculate safe price changes
        price_changes = []
        for i in range(1, len(prices)):
            change = safe_div(prices[i] - prices[i-1], prices[i-1])
            price_changes.append(max(min(change, 10), -10))  # Cap at 1000%
        
        # Calculate safe volume changes
        volume_changes = []
        for i in range(1, len(volumes)):
            change = safe_div(volumes[i] - volumes[i-1], volumes[i-1])
            volume_changes.append(max(min(change, 10), -10))  # Cap at 1000%
        
        # Calculate buying pressure safely
        buying_pressure = []
        for i in range(min(len(price_changes), len(volume_changes))):
            if price_changes[i] > 0:
                pressure = price_changes[i] * volume_changes[i]
                buying_pressure.append(max(min(pressure, 100), -100))  # Cap pressure
        
        return {
            'price_momentum': safe_div(np.mean(price_changes) if price_changes else 0, 1),
            'volume_momentum': safe_div(np.mean(volume_changes) if volume_changes else 0, 1),
            'buying_pressure': safe_div(np.mean(buying_pressure) if buying_pressure else 0, 1),
            'momentum_consistency': safe_div(np.sum(np.array(price_changes) > 0), len(price_changes)) if price_changes else 0,
            'price_volume_correlation': min(max(
                np.corrcoef(price_changes, volume_changes)[0,1] 
                if len(price_changes) == len(volume_changes) and len(price_changes) > 1 
                else 0, -1), 1)
        }
class ImprovedPumpDetector:
    def __init__(self):
        self.processor = ImprovedPumpDataProcessor()
        self.model = None

    def train(self, yes_dir: str = 'yes', no_dir: str = 'no'):
        """Train the model with improved robustness"""
        training_data = []

        # Process pump examples
        pump_examples = []
        for price_file, enhanced_file in self._find_pairs(yes_dir):
            features = self.processor.extract_features(price_file, enhanced_file)
            features['is_pump'] = 1
            pump_examples.append(features)

        # Process non-pump examples
        non_pump_examples = []
        for price_file, enhanced_file in self._find_pairs(no_dir):
            features = self.processor.extract_features(price_file, enhanced_file)
            features['is_pump'] = 0
            non_pump_examples.append(features)

        # Print dataset statistics
        print(f"Number of pump examples: {len(pump_examples)}")
        print(f"Number of non-pump examples: {len(non_pump_examples)}")

        # Balance dataset if needed
        min_samples = min(len(pump_examples), len(non_pump_examples))
        if min_samples < 10:
            print("Warning: Very small dataset. Consider adding more examples.")

        # Combine balanced datasets
        training_data = pump_examples[:min_samples] + non_pump_examples[:min_samples]

        if not training_data:
            raise ValueError("No training data found!")

        # Convert to DataFrame
        train_df = pd.DataFrame(training_data)

        # Add cross-validation with larger k for small datasets
        cv_folds = min(5, min_samples)  # Adjust folds based on sample size

        # Initialize PyCaret setup with corrected parameters
        self.model = setup(
            data=train_df,
            target='is_pump',
            session_id=123,
            normalize=True,
            feature_selection=True,
            fold=cv_folds,  # Adjusted fold count
            fold_strategy='stratifiedkfold'  # Corrected fold strategy
        )

        # Train with more robust parameters
        best_model = compare_models(
            n_select=1,
            fold=cv_folds,
            sort='AUC'  # More robust metric for imbalanced data
        )

        # More conservative tuning
        self.model = tune_model(
            best_model,
            n_iter=10,  # Reduced iterations to prevent overfitting
            optimize='AUC'
        )

        # Get feature importance
        importance = pd.DataFrame(get_feature_importance())
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))

        # Print cross-validation performance
        print("\nCross-validation performance:")
        return self.model
    def predict(self, price_file: str, enhanced_file: str) -> Dict:
        """Predict whether a token shows pump and dump patterns"""
        if not self.model:
            raise ValueError("Model not trained yet!")
            
        # Extract features
        features = self.processor.extract_features(price_file, enhanced_file)
        features_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = predict_model(self.model, data=features_df)
        
        return {
            'is_pump': bool(prediction['prediction_label'].iloc[0]),
            'confidence': float(prediction['prediction_score'].iloc[0]),
            'key_indicators': self._get_key_indicators(features)
        }
    
    def _get_key_indicators(self, features: Dict) -> Dict:
        """Extract key indicators that led to the prediction"""
        return {
            'price_patterns': {
                'price_increase': features['price_increase_pct'],
                'volatility': features['price_volatility_ratio'],
                'acceleration': features['price_acceleration']
            },
            'volume_patterns': {
                'spike_ratio': features['volume_spike_ratio'],
                'concentration': features['volume_concentration']
            },
            'account_patterns': {
                'growth': features['account_growth_ratio'],
                'concentration': features['concentration_spike_ratio']
            },
            'momentum': {
                'price_momentum': features['price_momentum'],
                'buying_pressure': features['buying_pressure']
            }
        }
    
    def _find_pairs(self, directory: str) -> List[tuple]:
        """Find matching price and enhanced data files"""
        files = os.listdir(directory)
        pairs = []
        
        price_files = {}
        enhanced_files = {}
        
        for file in files:
            if file.endswith('.json'):
                if file.startswith('price_history_'):
                    pattern = r'price_history_(.+)_(\d+_\d+).json'
                else:
                    pattern = r'enhanced_(.+)_(\d+_\d+).json'
                    
                match = re.match(pattern, file)
                if match:
                    token_address = match.group(1)
                    timestamp = match.group(2)
                    key = f"{token_address}_{timestamp}"
                    
                    if file.startswith('price_history_'):
                        price_files[key] = file
                    else:
                        enhanced_files[key] = file
        
        for key in price_files:
            if key in enhanced_files:
                pairs.append((
                    os.path.join(directory, price_files[key]),
                    os.path.join(directory, enhanced_files[key])
                ))
                
        return pairs


if __name__ == "__main__":
    try:
        # Initialize and train
        detector = ImprovedPumpDetector()
        detector.train()

        # Make prediction
        result = detector.predict('price_file.json', 'enhanced_file.json')
        
    except Exception as e:
        print(f"Error during training: {str(e)}")