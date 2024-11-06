import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

class DumpPredictor:
    def __init__(self, model_path="dump_predictor_model.joblib"):
        model_data = joblib.load(model_path)
        self.time_model = model_data['time_model']
        self.mcap_model = model_data['mcap_model']
        self.feature_columns = model_data['feature_columns']
        print(f"Required features: {self.feature_columns}")  # Debug print
    
    def prepare_features(self, enhanced_data, price_data):
        """Prepare features with all required columns"""
        features = []
        
        # Sort data by timestamp
        enhanced_data = sorted(enhanced_data, key=lambda x: x['timestamp'])
        price_data = sorted(price_data, key=lambda x: x['timestamp'])
        
        for window in enhanced_data:
            try:
                metrics = window['window_metrics']
                ts = window['timestamp']
                
                # Get relevant price data for this window
                relevant_prices = [p for p in price_data if p['timestamp'] <= ts + 20 and p['timestamp'] >= ts - 20]
                if not relevant_prices:
                    continue

                # Calculate market stats
                current_price = relevant_prices[-1]['price_in_usd']
                current_mcap = relevant_prices[-1]['market_cap']
                
                # Calculate price and volume changes
                price_changes = []
                volume_changes = []
                if len(relevant_prices) > 1:
                    for i in range(1, len(relevant_prices)):
                        if relevant_prices[i-1]['price_in_usd'] > 0:
                            price_changes.append((relevant_prices[i]['price_in_usd'] - relevant_prices[i-1]['price_in_usd']) / relevant_prices[i-1]['price_in_usd'])
                            volume_changes.append(relevant_prices[i]['sol_amount'] - relevant_prices[i-1]['sol_amount'])
                
                # Calculate metrics
                price_momentum = (relevant_prices[-1]['price_in_usd'] / relevant_prices[0]['price_in_usd'] - 1) if len(relevant_prices) > 1 else 0
                volume_momentum = sum(p['sol_amount'] for p in relevant_prices)
                price_acceleration = np.std(price_changes) if len(price_changes) > 1 else 0
                volume_acceleration = np.std(volume_changes) if len(volume_changes) > 1 else 0

                feature_dict = {
                    'timestamp': ts,
                    'current_price': current_price,
                    'current_mcap': current_mcap,
                    'price_momentum': price_momentum,
                    'volume_momentum': volume_momentum,
                    'valid_price_points': metrics['price_metrics']['valid_price_points'],
                    'sol_tx_count': metrics['sol_metrics']['sol_tx_count'],
                    'max_account_interactions': metrics['account_metrics'].get('max_account_interactions', 0),
                    'total_program_calls': metrics['program_metrics']['total_program_calls'],
                    'price_acceleration': price_acceleration,
                    'volume_acceleration': volume_acceleration,
                    'price_change': metrics['price_metrics']['price_change'],
                    'price_volatility': metrics['price_metrics']['price_volatility'],
                    'mcap_change': metrics['price_metrics']['mcap_change'],
                    'mcap_volatility': metrics['price_metrics']['mcap_volatility'],
                    'transfer_count': metrics['transaction_metrics']['transfer_count'],
                    'tx_density': metrics['transaction_metrics']['tx_density'],
                    'unique_accounts': metrics['transaction_metrics']['unique_accounts'],
                    'total_sol_volume': metrics['sol_metrics']['total_sol_volume'],
                    'sol_flow_volatility': metrics['sol_metrics']['sol_flow_volatility'],
                    'avg_sol_per_tx': metrics['sol_metrics']['avg_sol_per_tx'],
                    'interaction_concentration': metrics['account_metrics']['interaction_concentration'],
                    'avg_interactions_per_account': metrics['account_metrics']['avg_interactions_per_account'],
                    'unique_programs': metrics['program_metrics']['unique_programs'],
                    'avg_calls_per_program': metrics['program_metrics']['avg_calls_per_program']
                }
                
                features.append(feature_dict)
                
            except Exception as e:
                print(f"Error processing window at timestamp {ts}: {str(e)}")
                continue
                
        df = pd.DataFrame(features)
        print(f"Generated features: {df.columns.tolist()}")  # Debug print
        return df

    def predict(self, enhanced_data, price_data):
        """Predict dump characteristics"""
        try:
            # Prepare features from available data
            features_df = self.prepare_features(enhanced_data, price_data)
            
            if features_df.empty:
                return {
                    'seconds_until_dump': 0,
                    'dump_timestamp': 0,
                    'predicted_bottom_mcap': 0,
                    'expected_drop': 0,
                    'error': 'No valid features extracted'
                }
            
            # Ensure all required columns are present
            missing_columns = [col for col in self.feature_columns if col not in features_df.columns]
            if missing_columns:
                return {
                    'seconds_until_dump': 0,
                    'dump_timestamp': 0,
                    'predicted_bottom_mcap': 0,
                    'expected_drop': 0,
                    'error': f'Missing required features: {missing_columns}'
                }
            
            # Use the latest data point for prediction
            latest_features = features_df[self.feature_columns].iloc[-1:]
            
            # Make predictions
            seconds_to_dump = int(self.time_model.predict(latest_features)[0])
            bottom_mcap = float(self.mcap_model.predict(latest_features)[0])
            
            current_mcap = features_df.iloc[-1]['current_mcap']
            current_time = features_df.iloc[-1]['timestamp']
            
            return {
                'seconds_until_dump': seconds_to_dump,
                'dump_timestamp': current_time + seconds_to_dump,
                'predicted_bottom_mcap': bottom_mcap,
                'expected_drop': ((bottom_mcap - current_mcap) / current_mcap) * 100 if current_mcap != 0 else 0
            }
            
        except Exception as e:
            return {
                'seconds_until_dump': 0,
                'dump_timestamp': 0,
                'predicted_bottom_mcap': 0,
                'expected_drop': 0,
                'error': str(e)
            }

def predict_dump(enhanced_file, price_file, model_path="dump_predictor_model.joblib"):
    try:
        # Load the provided files
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        with open(price_file, 'r') as f:
            price_data = json.load(f)
            
        # Debug prints
        print(f"Enhanced data entries: {len(enhanced_data)}")
        print(f"Price data entries: {len(price_data)}")
        print(f"First enhanced timestamp: {enhanced_data[0]['timestamp']}")
        print(f"First price timestamp: {price_data[0]['timestamp']}")
        print(f"Last enhanced timestamp: {enhanced_data[-1]['timestamp']}")
        print(f"Last price timestamp: {price_data[-1]['timestamp']}")
        
        predictor = DumpPredictor(model_path)
        result = predictor.predict(enhanced_data, price_data)
        
        return result
        
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}'
        }

if __name__ == "__main__":
    result = predict_dump(
        enhanced_file="enhanced_8n6qNhEmzaikdk398Dd2GZ9XksPNwfPgBeYCb5b4XAjy_20241028_232515.json",
        price_file="price_history_8n6qNhEmzaikdk398Dd2GZ9XksPNwfPgBeYCb5b4XAjy_20241028_232515.json"
    )
    print(result)