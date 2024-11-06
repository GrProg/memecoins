import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import json
import joblib
from datetime import datetime
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Tuple, Any

class EnhancedDumpPredictor:
    def __init__(self, output_dir: str = "training_results"):
        self.time_model = None
        self.mcap_model = None
        self.feature_columns = None
        self.output_dir = output_dir
        self.training_metrics = {}
        os.makedirs(output_dir, exist_ok=True)
        self.training_log = []
        
    def log_message(self, message: str):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.training_log.append(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")

    def find_dumps(self, price_data: List[Dict], threshold: float = -0.15, window_size: int = 60) -> List[Dict]:
        """
        Identify dump events and capture pre-dump market caps
        threshold: percentage drop that defines a dump (-0.15 = 15% drop)
        window_size: seconds to look ahead for dump detection
        """
        dumps = []
        self.log_message(f"Analyzing price data for dumps (threshold: {threshold}, window: {window_size})")
        
        for i in range(len(price_data) - window_size):
            window = price_data[i:i + window_size]
            start_price = window[0]['price_in_usd']
            start_mcap = window[0]['market_cap']
            
            # Track the progression of prices
            for j in range(1, len(window)):
                current_price = window[j]['price_in_usd']
                price_change = (current_price - start_price) / start_price
                
                # If we detect a significant drop, this is a dump
                if price_change <= threshold:
                    # The pre-dump values are from the previous tick
                    pre_dump_idx = j - 1
                    pre_dump_mcap = window[pre_dump_idx]['market_cap']
                    
                    dumps.append({
                        'start_timestamp': window[0]['timestamp'],
                        'dump_timestamp': window[j]['timestamp'],
                        'pre_dump_mcap': pre_dump_mcap,  # Market cap right before dump
                        'time_to_dump': j,  # Number of ticks until dump
                        'pre_dump_price': window[pre_dump_idx]['price_in_usd'],
                        'initial_mcap': start_mcap,
                        'mcap_gain': ((pre_dump_mcap - start_mcap) / start_mcap) * 100
                    })
                    break
        
        self.log_message(f"Found {len(dumps)} dump events")
        return dumps

    def prepare_features(self, enhanced_data: List[Dict], price_data: List[Dict]) -> pd.DataFrame:
        """Prepare features with enhanced error handling and logging"""
        features = []
        self.log_message(f"Processing {len(enhanced_data)} windows of data")
        
        for window in enhanced_data:
            try:
                ts = window['timestamp']
                relevant_prices = [p for p in price_data if p['timestamp'] <= ts + 20 and p['timestamp'] >= ts - 20]
                
                if not relevant_prices:
                    continue
                    
                window_metrics = window['window_metrics']
                
                # Calculate price and volume changes
                price_changes = []
                volume_changes = []
                if len(relevant_prices) > 1:
                    for i in range(1, len(relevant_prices)):
                        if relevant_prices[i-1]['price_in_usd'] > 0:
                            price_changes.append(
                                (relevant_prices[i]['price_in_usd'] - relevant_prices[i-1]['price_in_usd']) 
                                / relevant_prices[i-1]['price_in_usd']
                            )
                        volume_changes.append(relevant_prices[i].get('sol_amount', 0) - relevant_prices[i-1].get('sol_amount', 0))
                
                feature_dict = {
                    'timestamp': ts,
                    'current_price': relevant_prices[-1]['price_in_usd'],
                    'current_mcap': relevant_prices[-1]['market_cap'],
                    'price_momentum': (relevant_prices[-1]['price_in_usd'] / relevant_prices[0]['price_in_usd'] - 1) if len(relevant_prices) > 1 else 0,
                    'volume_momentum': sum(p.get('sol_amount', 0) for p in relevant_prices),
                    'valid_price_points': window_metrics['price_metrics']['valid_price_points'],
                    'sol_tx_count': window_metrics['sol_metrics']['sol_tx_count'],
                    'max_account_interactions': window_metrics['account_metrics'].get('max_account_interactions', 0),
                    'total_program_calls': window_metrics['program_metrics']['total_program_calls'],
                    'price_acceleration': np.std(price_changes) if len(price_changes) > 1 else 0,
                    'volume_acceleration': np.std(volume_changes) if len(volume_changes) > 1 else 0,
                    'price_change': window_metrics['price_metrics']['price_change'],
                    'price_volatility': window_metrics['price_metrics']['price_volatility'],
                    'mcap_change': window_metrics['price_metrics']['mcap_change'],
                    'mcap_volatility': window_metrics['price_metrics']['mcap_volatility'],
                    'transfer_count': window_metrics['transaction_metrics']['transfer_count'],
                    'tx_density': window_metrics['transaction_metrics']['tx_density'],
                    'unique_accounts': window_metrics['transaction_metrics']['unique_accounts'],
                    'total_sol_volume': window_metrics['sol_metrics']['total_sol_volume'],
                    'sol_flow_volatility': window_metrics['sol_metrics']['sol_flow_volatility'],
                    'avg_sol_per_tx': window_metrics['sol_metrics']['avg_sol_per_tx'],
                    'interaction_concentration': window_metrics['account_metrics']['interaction_concentration'],
                    'avg_interactions_per_account': window_metrics['account_metrics']['avg_interactions_per_account'],
                    'unique_programs': window_metrics['program_metrics']['unique_programs'],
                    'avg_calls_per_program': window_metrics['program_metrics']['avg_calls_per_program']
                }
                
                features.append(feature_dict)
                
            except Exception as e:
                self.log_message(f"Error processing window at timestamp {ts}: {str(e)}")
                continue
                
        return pd.DataFrame(features)

    def prepare_training_data(self, enhanced_data: List[Dict], price_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare features and labels for training"""
        features_df = self.prepare_features(enhanced_data, price_data)
        
        # Find dumps and pre-dump market caps
        dumps = self.find_dumps(price_data)
        
        # Prepare labels
        labels = []
        timestamps = features_df['timestamp'].values
        
        for ts in timestamps:
            future_dumps = [d for d in dumps if d['start_timestamp'] > ts]
            
            if future_dumps:
                next_dump = min(future_dumps, key=lambda x: x['start_timestamp'])
                labels.append([
                    next_dump['dump_timestamp'] - ts,  # time until dump
                    next_dump['pre_dump_mcap']  # market cap right before dump
                ])
            else:
                labels.append([np.nan, np.nan])
        
        labels = np.array(labels)
        
        # Remove rows with NaN labels
        valid_mask = ~np.isnan(labels).any(axis=1)
        features_df = features_df[valid_mask]
        labels = labels[valid_mask]
        
        self.feature_columns = [col for col in features_df.columns if col != 'timestamp']
        
        return features_df[self.feature_columns], labels

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: np.ndarray, target_name: str) -> Dict:
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mape': mean_absolute_percentage_error(y_test, predictions) * 100,  # Convert to percentage
            'r2': r2_score(y_test, predictions)
        }
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{target_name} - Actual vs Predicted')
        plt.savefig(os.path.join(self.output_dir, f'{target_name.lower()}_predictions.png'))
        plt.close()
        
        return metrics

    def plot_feature_importance(self, model, target_name: str):
        """Plot feature importance"""
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance.head(15))
        plt.title(f'Top 15 Important Features for {target_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{target_name.lower()}_importance.png'))
        plt.close()
        
        return importance

    def train(self, all_enhanced_data: List[List[Dict]], all_price_data: List[List[Dict]]):
        """Train models with comprehensive evaluation"""
        all_features = []
        all_labels = []
        
        self.log_message("Starting training process")
        
        # Process each pair of data
        for enhanced_data, price_data in zip(all_enhanced_data, all_price_data):
            X, y = self.prepare_training_data(enhanced_data, price_data)
            all_features.append(X)
            all_labels.append(y)
        
        # Combine all training data
        X = pd.concat(all_features, axis=0) if all_features else pd.DataFrame()
        y = np.vstack(all_labels) if all_labels else np.array([])
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid training data found")
            
        self.log_message(f"Total training samples: {len(X)}")
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train time model
        self.log_message("Training time prediction model")
        self.time_model = LGBMRegressor(
            objective='regression',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1
        )
        self.time_model.fit(X_train, y_train[:, 0])
        time_metrics = self.evaluate_model(self.time_model, X_test, y_test[:, 0], 'Time to Dump')
        time_importance = self.plot_feature_importance(self.time_model, 'Time to Dump')
        
        # Train pre-dump mcap model
        self.log_message("Training pre-dump market cap prediction model")
        self.mcap_model = LGBMRegressor(
            objective='regression',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1
        )
        self.mcap_model.fit(X_train, y_train[:, 1])
        mcap_metrics = self.evaluate_model(self.mcap_model, X_test, y_test[:, 1], 'Pre-Dump Market Cap')
        mcap_importance = self.plot_feature_importance(self.mcap_model, 'Pre-Dump Market Cap')
        
        # Save training metrics
        self.training_metrics = {
            'time_model_metrics': time_metrics,
            'mcap_model_metrics': mcap_metrics,
            'feature_importance': {
                'time_model': time_importance.to_dict(orient='records'),
                'mcap_model': mcap_importance.to_dict(orient='records')
            },
            'training_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': len(self.feature_columns)
        }
        
        # Save training report
        self.save_training_report()
        
        return self

    def save_model(self, filepath: str):
        """Save the trained model and related data"""
        model_data = {
            'time_model': self.time_model,
            'mcap_model': self.mcap_model,
            'feature_columns': self.feature_columns,
            'training_metrics': self.training_metrics,
            'training_log': self.training_log
        }
        joblib.dump(model_data, filepath)
        self.log_message(f"Model saved to {filepath}")

    def save_training_report(self):
        """Save detailed training report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.training_metrics,
            'training_log': self.training_log
        }
        
        report_path = os.path.join(self.output_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log_message(f"Training report saved to {report_path}")

def train_model_from_folder(yes_folder: str = "yes", 
                          output_dir: str = "training_results",
                          model_output_path: str = "dump_predictor_model.joblib"):
    """Train model from data folder with detailed logging"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to both file and console
    log_file = os.path.join(output_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting model training process")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Model will be saved to: {model_output_path}")
    
    # Get training files
    enhanced_files = glob.glob(os.path.join(yes_folder, "enhanced_*.json"))
    logging.info(f"Found {len(enhanced_files)} potential training files")
    
    all_enhanced_data = []
    all_price_data = []
    processed_pairs = 0
    
    for enhanced_file in enhanced_files:
        try:
            # Extract token identifier from filename
            base_name = os.path.basename(enhanced_file)
            token_id = base_name.split('_')[1]
            timestamp = '_'.join(base_name.split('_')[2:]).replace('.json', '')
            
            # Find matching price history file
            price_file = os.path.join(yes_folder, f"price_history_{token_id}_{timestamp}.json")
            
            if os.path.exists(price_file):
                logging.info(f"\nProcessing pair {processed_pairs + 1}/{len(enhanced_files)}:")
                logging.info(f"Enhanced: {base_name}")
                logging.info(f"Price: price_history_{token_id}_{timestamp}.json")
                
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f)
                with open(price_file, 'r') as f:
                    price_data = json.load(f)
                
                logging.info(f"Enhanced data entries: {len(enhanced_data)}")
                logging.info(f"Price data entries: {len(price_data)}")
                
                all_enhanced_data.append(enhanced_data)
                all_price_data.append(price_data)
                processed_pairs += 1
                logging.info("✓ Successfully loaded data pair")
            else:
                logging.warning(f"✗ No matching price file found for {base_name}")
                
        except Exception as e:
            logging.error(f"Error processing {enhanced_file}: {str(e)}")
            continue
    
    if not all_enhanced_data:
        logging.error("No valid data pairs found for training!")
        return
    
    logging.info(f"\nSuccessfully loaded {processed_pairs} data pairs")
    
    try:
        # Initialize and train model
        predictor = EnhancedDumpPredictor(output_dir=output_dir)
        predictor.train(all_enhanced_data, all_price_data)
        
        # Save model
        predictor.save_model(model_output_path)
        logging.info(f"\nModel trained and saved to {model_output_path}")
        logging.info(f"Training results and visualizations saved to {output_dir}")
        
        # Print summary metrics
        metrics = predictor.training_metrics
        logging.info("\nTraining Results Summary:")
        logging.info("Time Model Performance:")
        logging.info(f"  RMSE: {metrics['time_model_metrics']['rmse']:.2f} seconds")
        logging.info(f"  MAPE: {metrics['time_model_metrics']['mape']:.2f}%")
        logging.info(f"  R²: {metrics['time_model_metrics']['r2']:.3f}")
        
        logging.info("\nPre-Dump Market Cap Model Performance:")
        logging.info(f"  RMSE: {metrics['mcap_model_metrics']['rmse']:.2f} USD")
        logging.info(f"  MAPE: {metrics['mcap_model_metrics']['mape']:.2f}%")
        logging.info(f"  R²: {metrics['mcap_model_metrics']['r2']:.3f}")
        
        # Log feature importance summary
        logging.info("\nTop 5 Important Features for Time Prediction:")
        for feat in predictor.training_metrics['feature_importance']['time_model'][:5]:
            logging.info(f"  {feat['feature']}: {feat['importance']:.4f}")
            
        logging.info("\nTop 5 Important Features for MCap Prediction:")
        for feat in predictor.training_metrics['feature_importance']['mcap_model'][:5]:
            logging.info(f"  {feat['feature']}: {feat['importance']:.4f}")
        
    except Exception as e:
        logging.error(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model_from_folder()