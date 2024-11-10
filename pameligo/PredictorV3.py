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
        
    def extract_features(self, price_data: pd.DataFrame, enhanced_data: pd.DataFrame) -> Dict:
        """Extract features from current market state"""
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

    def predict(self, price_file: str, enhanced_file: str) -> Dict:
        """Make prediction from price and enhanced data files"""
        # Load data
        with open(price_file, 'r') as f:
            price_data = pd.DataFrame(json.load(f))
        with open(enhanced_file, 'r') as f:
            enhanced_data = pd.DataFrame(json.load(f))
            
        # Sort by timestamp
        price_data = price_data.sort_values('timestamp')
        enhanced_data = enhanced_data.sort_values('timestamp')
        
        # Get current stats
        current_mcap = price_data.iloc[-1]['market_cap']
        current_price = price_data.iloc[-1]['price_in_usd']
        target_mcap = current_mcap + self.target_mcap_increase
        
        # Extract features
        features = self.extract_features(price_data, enhanced_data)
        features_df = pd.DataFrame([features])
        
        # Make prediction
        probability = self.model.predict_proba(features_df)[0][1]
        prediction = self.model.predict(features_df)[0]
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_market_cap': current_mcap,
            'current_price_usd': current_price,
            'target_market_cap': target_mcap,
            'predicted_outcome': 'PUMP' if prediction == 1 else 'NO PUMP',
            'confidence': probability,
            'key_metrics': {
                'price_volatility': features['price_volatility'],
                'volume_acceleration': features['volume_acceleration'],
                'tx_density': features['tx_density'],
                'unique_accounts': features['unique_accounts']
            }
        }

def predict_market_cap(price_file: str, enhanced_file: str):
    """Helper function to make and print prediction"""
    predictor = MarketCapPredictor()
    result = predictor.predict(price_file, enhanced_file)
    
    print("\n=== Market Cap Prediction ===")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Current Market Cap: ${result['current_market_cap']:,.2f}")
    print(f"Target (+$10k): ${result['target_market_cap']:,.2f}")
    print(f"Current Price: ${result['current_price_usd']:,.6f}")
    print("\nPrediction:")
    print(f"Outcome: {result['predicted_outcome']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    print("\nKey Metrics:")
    print(f"Price Volatility: {result['key_metrics']['price_volatility']:.4f}")
    print(f"Volume Acceleration: {result['key_metrics']['volume_acceleration']:.4f}")
    print(f"Transaction Density: {result['key_metrics']['tx_density']:.2f}")
    print(f"Unique Accounts: {result['key_metrics']['unique_accounts']}")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python market_predictor.py price_file.json enhanced_file.json")
        sys.exit(1)
        
    predict_market_cap(sys.argv[2], sys.argv[1])