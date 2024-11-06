import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime

class DumpPredictor:
    def __init__(self, model_path="dump_detector.joblib"):
        """Load trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.mcap_threshold = model_data['mcap_threshold']
        
    def safe_calculate(self, operation, default=0.0):
        """Safely perform calculations handling div by zero and infinity"""
        try:
            result = operation()
            if np.isfinite(result):
                return result
            return default
        except:
            return default
            
    def extract_features(self, enhanced_data, price_data):
        """Extract features with enhanced safety checks"""
        # Sort both datasets by timestamp ascending
        sorted_enhanced = sorted(enhanced_data, key=lambda x: x['timestamp'])
        sorted_prices = sorted(price_data, key=lambda x: x['timestamp'])
        
        if not sorted_enhanced or not sorted_prices:
            raise ValueError("Empty input data")
            
        features = []
        
        # Calculate time-based metrics safely
        total_time_span = max(1, sorted_prices[-1]['timestamp'] - sorted_prices[0]['timestamp'])
        initial_mcap = sorted_prices[0]['market_cap']
        current_mcap = sorted_prices[-1]['market_cap']
        
        # Safe growth rate calculation
        mcap_growth_rate = self.safe_calculate(
            lambda: (current_mcap - initial_mcap) / total_time_span
        )
        
        for window in sorted_enhanced:
            try:
                window_time = window['timestamp']
                metrics = window['window_metrics']
                
                # Get price data for this window with safety check
                window_prices = [p for p in sorted_prices if abs(p['timestamp'] - window_time) <= 20]
                if len(window_prices) < 2:
                    continue
                    
                window_start_mcap = window_prices[0]['market_cap']
                window_end_mcap = window_prices[-1]['market_cap']
                window_time_span = max(1, window_prices[-1]['timestamp'] - window_prices[0]['timestamp'])
                
                # Safe feature calculations with defaults
                window_features = {
                    # Price and volatility metrics
                    'price_volatility': float(metrics['price_metrics'].get('price_volatility', 0)),
                    'mcap_volatility': float(metrics['price_metrics'].get('mcap_volatility', 0)),
                    'price_change': float(metrics['price_metrics'].get('price_change', 0)),
                    
                    # Transaction metrics
                    'tx_density': float(metrics['transaction_metrics'].get('tx_density', 0)),
                    'transfer_count': int(metrics['transaction_metrics'].get('transfer_count', 0)),
                    'unique_accounts': int(metrics['transaction_metrics'].get('unique_accounts', 0)),
                    
                    # Volume metrics
                    'total_sol_volume': float(metrics['sol_metrics'].get('total_sol_volume', 0)),
                    'sol_flow_volatility': float(metrics['sol_metrics'].get('sol_flow_volatility', 0)),
                    'avg_sol_per_tx': float(metrics['sol_metrics'].get('avg_sol_per_tx', 0)),
                    
                    # Account metrics
                    'interaction_concentration': float(metrics['account_metrics'].get('interaction_concentration', 0)),
                    'avg_interactions': float(metrics['account_metrics'].get('avg_interactions_per_account', 0)),
                    
                    # Program metrics
                    'unique_programs': int(metrics['program_metrics'].get('unique_programs', 0)),
                    'avg_calls_per_program': float(metrics['program_metrics'].get('avg_calls_per_program', 0)),
                    
                    # Time-based metrics (safely calculated)
                    'mcap_growth_rate': mcap_growth_rate,
                    
                    'time_to_threshold': self.safe_calculate(
                        lambda: (55000 - current_mcap) / mcap_growth_rate if mcap_growth_rate > 0 else 1e6,
                        default=1e6
                    ),
                    
                    'acceleration': self.safe_calculate(
                        lambda: (window_end_mcap - window_start_mcap) / window_time_span
                    ),
                    
                    'momentum_score': self.safe_calculate(
                        lambda: mcap_growth_rate * metrics['sol_metrics'].get('total_sol_volume', 0)
                    ),
                    
                    'buy_pressure': self.safe_calculate(
                        lambda: (metrics['sol_metrics'].get('total_sol_volume', 0) / window_time_span) * 
                               (window_end_mcap / max(1, window_start_mcap))
                    ),
                    
                    'volatility_trend': self.safe_calculate(
                        lambda: metrics['price_metrics'].get('price_volatility', 0) / 
                               max(1, window_time - sorted_prices[0]['timestamp'])
                    ),
                    
                    'volume_acceleration': self.safe_calculate(
                        lambda: metrics['sol_metrics'].get('total_sol_volume', 0) / 
                               max(1, window_time - sorted_prices[0]['timestamp'])
                    ),
                    
                    'price_efficiency': self.safe_calculate(
                        lambda: abs(window_end_mcap - window_start_mcap) / 
                               max(0.001, metrics['sol_metrics'].get('total_sol_volume', 0))
                    ),
                    
                    'time_weighted_tx_density': self.safe_calculate(
                        lambda: metrics['transaction_metrics'].get('tx_density', 0) * 
                               (window_time - sorted_prices[0]['timestamp'])
                    ),
                    
                    'relative_progress': self.safe_calculate(
                        lambda: (current_mcap - initial_mcap) / max(1, 55000 - initial_mcap)
                    )
                }
                
                # Clip any extreme values to reasonable ranges
                for key, value in window_features.items():
                    if isinstance(value, (int, float)):
                        window_features[key] = np.clip(value, -1e6, 1e6)
                
                features.append(window_features)
                
            except Exception as e:
                print(f"Error processing window at {window_time}: {str(e)}")
                continue
        
        if not features:
            raise ValueError("No valid features could be extracted")
            
        return pd.DataFrame(features)

    def predict(self, enhanced_data, price_data):
        """Make prediction with comprehensive error handling"""
        try:
            # Extract features
            features = self.extract_features(enhanced_data, price_data)
            
            # Ensure we have all required features
            missing_features = set(self.feature_columns) - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Reorder columns to match training data
            features = features[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate current metrics
            current_mcap = sorted(price_data, key=lambda x: x['timestamp'])[-1]['market_cap']
            mcap_velocity = features['mcap_growth_rate'].iloc[-1]
            
            # Prepare results
            results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_mcap': float(current_mcap),
                'will_dump': bool(predictions[-1]),
                'confidence': float(probabilities[-1]),
                'mcap_to_threshold': float(self.mcap_threshold - current_mcap),
                'estimated_minutes_to_threshold': float(
                    self.safe_calculate(
                        lambda: (self.mcap_threshold - current_mcap) / (mcap_velocity * 60) 
                        if mcap_velocity > 0 else float('inf'),
                        default=float('inf')
                    )
                ),
                'windows_analyzed': len(enhanced_data),
                'dump_signals': int(sum(predictions)),
                'average_confidence': float(probabilities.mean()),
                'momentum_indicators': {
                    'mcap_velocity': float(mcap_velocity),
                    'buy_pressure': float(features['buy_pressure'].iloc[-1]),
                    'volume_acceleration': float(features['volume_acceleration'].iloc[-1]),
                    'price_efficiency': float(features['price_efficiency'].iloc[-1])
                }
            }
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'failed'
            }

def predict_dump(enhanced_file, price_file):
    """Main prediction function with error handling"""
    try:
        # Load data
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        with open(price_file, 'r') as f:
            price_data = json.load(f)
            
        # Basic data validation
        if not enhanced_data or not price_data:
            raise ValueError("Empty input data")
            
        # Make prediction
        predictor = DumpPredictor()
        results = predictor.predict(enhanced_data, price_data)
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return results
            
        print(f"Timestamp: {results['timestamp']}")
        print(f"Current Market Cap: ${results['current_mcap']:,.2f}")
        print(f"Distance to 55k: ${results['mcap_to_threshold']:,.2f}")
        print(f"Prediction: {'YES' if results['will_dump'] else 'NO'}")
        print(f"Confidence: {results['confidence']:.1%}")
        print(f"\nAnalysis Details:")
        print(f"Windows analyzed: {results['windows_analyzed']}")
        print(f"Dump signals detected: {results['dump_signals']}")
        print(f"Average confidence: {results['average_confidence']:.1%}")
        
        return results
        
    except Exception as e:
        print(f"Critical Error: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python predictor.py enhanced_file.json price_file.json")
    else:
        predict_dump(sys.argv[1], sys.argv[2])