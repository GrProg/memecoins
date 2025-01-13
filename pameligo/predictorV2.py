import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class PumpPredictorV2:
    def __init__(self, model_path="pump_detector_v3.joblib"):
        """Initialize predictor with trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.thresholds = model_data['thresholds']
        
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
        
        price_sequence = [w['window_metrics']['price_metrics']['price_end'] for w in windows[-lookback:]]
        volume_sequence = [w['window_metrics']['sol_metrics']['total_sol_volume'] for w in windows[-lookback:]]
        
        changes.update({
            'price_volatility': np.std(price_sequence) / np.mean(price_sequence) if len(price_sequence) > 1 else 0,
            'volume_volatility': np.std(volume_sequence) / np.mean(volume_sequence) if len(volume_sequence) > 1 else 0
        })
        
        return changes

    def prepare_features(self, windows: List[Dict]) -> Dict[str, float]:
        """Extract features from window sequence"""
        if not windows:
            return {}
            
        current = windows[-1]['window_metrics']
        
        features = {
            'price': current['price_metrics']['price_end'],
            'volume': current['sol_metrics']['total_sol_volume'],
            'tx_count': current['transaction_metrics']['transfer_count'],
            'unique_accounts': current['transaction_metrics']['unique_accounts'],
            'avg_tx_size': current['sol_metrics']['avg_sol_per_tx'],
            'concentration': current['account_metrics']['interaction_concentration']
        }
        
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

    def predict(self, enhanced_data: List[Dict], price_data: List[Dict]) -> Dict:
        """Make predictions for all 10-second windows"""
        try:
            predictions = {"predictions": {}}
            
            # Sort data by timestamp
            enhanced_data = sorted(enhanced_data, key=lambda x: x['timestamp'])
            start_time = enhanced_data[0]['timestamp']
            end_time = enhanced_data[-1]['timestamp']
            
            # Calculate total duration and number of 10-second windows
            total_duration = end_time - start_time
            window_count = (total_duration // 10) + 1
            
            # Analyze each window
            for window_num in range(window_count):
                window_end = start_time + (window_num + 1) * 10
                window_data = [w for w in enhanced_data if w['timestamp'] <= window_end]
                
                if window_data:
                    features = self.prepare_features(window_data)
                    if features:
                        # Prepare features for model
                        features_df = pd.DataFrame([features])
                        features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
                        X_scaled = self.scaler.transform(features_df)
                        
                        # Make prediction
                        prob = self.model.predict_proba(X_scaled)[0][1]
                        pred = 1 if prob >= 0.5 else 0
                        
                        window_key = f"0-{(window_num + 1) * 10}"
                        predictions["predictions"][window_key] = {
                            "prediction": int(pred),  # 1 for pump, 0 for no pump
                            "confidence": float(prob),
                            "timestamp": window_data[-1]['timestamp']
                        }
            
            return predictions
            
        except Exception as e:
            return {"error": str(e)}

def predict_pump(enhanced_file: str, price_file: str) -> Dict:
    """Main prediction function"""
    try:
        # Load data files
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        with open(price_file, 'r') as f:
            price_data = json.load(f)
            
        # Initialize predictor and make predictions
        predictor = PumpPredictorV2()
        results = predictor.predict(enhanced_data, price_data)
        
        # Print formatted results
        print("\nPrediction Results:")
        print("-" * 50)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return results
            
        for window, data in sorted(results["predictions"].items()):
            print(f"\nTime Window {window}:")
            print(f"Prediction: {data['prediction']}")  # 1 for pump, 0 for no pump
            print(f"Confidence: {data['confidence']:.2%}")
            print(f"Timestamp: {datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results
        
    except Exception as e:
        error_result = {"error": str(e)}
        print(f"\nError: {str(e)}")
        return error_result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python PredictorV2.py enhanced_file.json price_file.json")
    else:
        predict_pump(sys.argv[1], sys.argv[2])