import joblib
import pandas as pd
import json
from typing import Dict, List
import numpy as np
from datetime import datetime

class MarketCapPredictor:
    def __init__(self, model_path="mcap_prediction_model.joblib"):
        """Load the trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.target_mcap_increase = model_data['target_mcap_increase']
        self.window_size = model_data['window_size']
        
    def create_windows(self, price_data: pd.DataFrame, enhanced_data: pd.DataFrame) -> List[Dict]:
        """Create time windows from the data."""
        windows = []
        start_time = price_data['timestamp'].min()
        end_time = price_data['timestamp'].max()
        
        # Print data structure for debugging
        print("\nPrice data columns:", price_data.columns.tolist())
        
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
            
            # Skip if no transactions
            if len(window_prices) == 0:
                current_time += self.window_size
                continue
                
            # Create window dict with price from whatever column contains it
            window = {
                'start_time': current_time,
                'end_time': window_end,
                'price_data': window_prices,
                'enhanced_data': window_enhanced,
            }
            
            windows.append(window)
            current_time += self.window_size
            
        return windows

    def extract_features(self, price_data: pd.DataFrame, enhanced_data: pd.DataFrame) -> Dict:
        """Extract features from current market state"""
        # Handle different price column names
        price_col = None
        if 'price_in_usd' in price_data.columns:
            price_col = 'price_in_usd'
        elif 'price' in price_data.columns:
            price_col = 'price'
        
        if not price_col:
            raise ValueError(f"No price column found. Available columns: {price_data.columns.tolist()}")
        
        # Market Cap and Price Features
        market_features = {
            'mcap_current': price_data['market_cap'].iloc[-1],
            'mcap_mean': price_data['market_cap'].mean(),
            'mcap_std': price_data['market_cap'].std(),
            'mcap_min': price_data['market_cap'].min(),
            'mcap_max': price_data['market_cap'].max(),
            'mcap_change': price_data['market_cap'].pct_change().mean(),
            'mcap_acceleration': price_data['market_cap'].diff().diff().mean(),
            'price_change': price_data[price_col].pct_change().mean(),
            'price_volatility': price_data[price_col].std() / price_data[price_col].mean()
        }
        
        # Volume Features
        volume_features = {
            'volume_mean': price_data['sol_amount'].mean(),
            'volume_total': price_data['sol_amount'].sum(),
            'tx_count': len(price_data),
            'avg_tx_size': price_data['sol_amount'].mean(),
            'volume_acceleration': price_data['sol_amount'].diff().diff().mean()
        }
        
        # Enhanced metrics
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

    def analyze_window(self, window: Dict, future_price_data: pd.DataFrame) -> Dict:
        """Analyze a single time window and make prediction"""
        # Extract features
        features = self.extract_features(window['price_data'], window['enhanced_data'])
        features_df = pd.DataFrame([features])
        
        # Make prediction
        probability = self.model.predict_proba(features_df)[0][1]
        prediction = self.model.predict(features_df)[0]
        
        # Get current stats
        current_mcap = window['price_data'].iloc[-1]['market_cap']
        # Get price from whatever column contains it
        price_col = 'price_in_usd' if 'price_in_usd' in window['price_data'].columns else 'price'
        current_price = window['price_data'].iloc[-1][price_col]
        target_mcap = current_mcap + self.target_mcap_increase
        
        # Get actual outcome if we have future data
        future_mcaps = future_price_data[
            future_price_data['timestamp'] > window['end_time']
        ]['market_cap'].tolist()
        
        actual_outcome = None
        max_future_mcap = None
        if future_mcaps:
            max_future_mcap = max(future_mcaps)
            actual_outcome = 'PUMP' if max_future_mcap >= target_mcap else 'NO PUMP'
        
        return {
            'window_start': datetime.fromtimestamp(window['start_time']).strftime('%Y-%m-%d %H:%M:%S'),
            'window_end': datetime.fromtimestamp(window['end_time']).strftime('%Y-%m-%d %H:%M:%S'),
            'current_market_cap': current_mcap,
            'current_price_usd': current_price,
            'target_market_cap': target_mcap,
            'predicted_outcome': 'PUMP' if prediction == 1 else 'NO PUMP',
            'confidence': probability,
            'actual_outcome': actual_outcome,
            'max_future_mcap': max_future_mcap,
            'key_metrics': {
                'price_volatility': features['price_volatility'],
                'volume_acceleration': features['volume_acceleration'],
                'tx_density': features['tx_density'],
                'unique_accounts': features['unique_accounts']
            }
        }

    def predict(self, price_file: str, enhanced_file: str) -> List[Dict]:
        """Make predictions for all time windows"""
        # Load data
        with open(price_file, 'r') as f:
            price_data = pd.DataFrame(json.load(f))
        with open(enhanced_file, 'r') as f:
            enhanced_data = pd.DataFrame(json.load(f))
            
        # Sort by timestamp
        price_data = price_data.sort_values('timestamp')
        enhanced_data = enhanced_data.sort_values('timestamp')
        
        print("Loading data completed...")
        print("Price data shape:", price_data.shape)
        print("Enhanced data shape:", enhanced_data.shape)
        
        # Create windows
        windows = self.create_windows(price_data, enhanced_data)
        print(f"Created {len(windows)} windows")
        
        # Analyze each window
        results = []
        for i, window in enumerate(windows, 1):
            try:
                result = self.analyze_window(window, price_data)
                results.append(result)
                if i % 10 == 0:
                    print(f"Analyzed {i} windows...")
            except Exception as e:
                print(f"Error analyzing window {i}: {str(e)}")
                continue
            
        return results

def predict_market_cap(price_file: str, enhanced_file: str):
    """Helper function to make and print predictions"""
    predictor = MarketCapPredictor()
    results = predictor.predict(price_file, enhanced_file)
    
    print(f"\n=== Market Cap Predictions ({len(results)} windows) ===")
    
    correct_predictions = 0
    total_predictions_with_outcome = 0
    
    for i, result in enumerate(results, 1):
        print(f"\nWindow {i}:")
        print(f"Time: {result['window_start']} to {result['window_end']}")
        print(f"Market Cap: ${result['current_market_cap']:,.2f}")
        print(f"Prediction: {result['predicted_outcome']} ({result['confidence']:.1%})")
        
        if result['actual_outcome']:
            print(f"Actual Outcome: {result['actual_outcome']}")
            print(f"Max Future MCap: ${result['max_future_mcap']:,.2f}")
            total_predictions_with_outcome += 1
            if result['predicted_outcome'] == result['actual_outcome']:
                correct_predictions += 1
    

    
    if total_predictions_with_outcome > 0:
        accuracy = correct_predictions / total_predictions_with_outcome * 100
        print(f"Prediction Accuracy: {accuracy:.1f}%")
    
    # Get average metrics across all windows
    avg_metrics = {
        'price_volatility': np.mean([r['key_metrics']['price_volatility'] for r in results]),
        'volume_acceleration': np.mean([r['key_metrics']['volume_acceleration'] for r in results]),
        'tx_density': np.mean([r['key_metrics']['tx_density'] for r in results]),
        'unique_accounts': np.mean([r['key_metrics']['unique_accounts'] for r in results])
    }
    

    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python market_predictor.py price_file.json enhanced_file.json")
        sys.exit(1)
        
    predict_market_cap(sys.argv[2], sys.argv[1])