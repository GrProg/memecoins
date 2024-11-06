import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import re

class TokenClassifier:
    def __init__(self, model_path='pump_detector_final'):
        """Initialize with trained model"""
        self.model = load_model(model_path)
    
    def find_and_sort_files(self, directory: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Find matching pairs and identify unmatched files"""
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist!")
            return [], []
            
        files = os.listdir(directory)
        if not files:
            print(f"Warning: Directory '{directory}' is empty!")
            return [], []
            
        print(f"\nScanning directory: {directory}")
        print(f"Total files found: {len(files)}")
        
        enhanced_files = {}
        price_files = {}
        unmatched_files = []
        
        enhanced_count = 0
        price_count = 0
        
        # Sort files into categories
        for file in files:
            if not file.endswith('.json'):
                continue
                
            if file.startswith('enhanced_'):
                pattern = r'enhanced_(.+)_(\d+_\d+).json'
                enhanced_count += 1
            else:
                pattern = r'price_history_(.+)_(\d+_\d+).json'
                price_count += 1
            
            match = re.match(pattern, file)
            if match:
                token_address = match.group(1)
                timestamp = match.group(2)
                key = f"{token_address}_{timestamp}"
                
                if file.startswith('enhanced_'):
                    enhanced_files[key] = file
                else:
                    price_files[key] = file
        
        # Find pairs and unmatched files
        pairs = []
        for key in price_files:
            if key in enhanced_files:
                pairs.append((
                    os.path.join(directory, price_files[key]),
                    os.path.join(directory, enhanced_files[key])
                ))
                print(f"Found matching pair for token {key.split('_')[0][:8]}...")
            else:
                unmatched_files.append(price_files[key])
                print(f"No enhanced file for price history: {price_files[key]}")
        
        # Summary
        print(f"\nFile Summary:")
        print(f"Enhanced files found: {enhanced_count}")
        print(f"Price history files found: {price_count}")
        print(f"Complete pairs found: {len(pairs)}")
        print(f"Unmatched price files: {len(unmatched_files)}")
        
        return pairs, unmatched_files
    
    def extract_features(self, price_file: str, enhanced_file: str) -> Dict:
        """Extract features from both files"""
        with open(price_file, 'r') as f:
            price_data = pd.DataFrame(json.load(f))
        with open(enhanced_file, 'r') as f:
            enhanced_data = pd.DataFrame(json.load(f))
            
        # Price-based features
        price_features = {
            'max_market_cap': price_data['market_cap'].max(),
            'min_market_cap': price_data['market_cap'].min(),
            'price_volatility': price_data['price_in_usd'].std() / price_data['price_in_usd'].mean(),
            'mcap_range': price_data['market_cap'].max() - price_data['market_cap'].min(),
            'price_acceleration': np.diff(price_data['price_in_usd']).std(),
        }
        
        # Enhanced data features
        if not enhanced_data.empty:
            window_metrics = [row['window_metrics'] for _, row in enhanced_data.iterrows()]
            
            tx_metrics = [w['transaction_metrics'] for w in window_metrics]
            sol_metrics = [w['sol_metrics'] for w in window_metrics]
            account_metrics = [w['account_metrics'] for w in window_metrics]
            
            enhanced_features = {
                'max_tx_density': max(w['tx_density'] for w in tx_metrics),
                'avg_tx_density': np.mean([w['tx_density'] for w in tx_metrics]),
                'peak_volume': max(w['total_sol_volume'] for w in sol_metrics),
                'volume_volatility': np.mean([w['sol_flow_volatility'] for w in sol_metrics]),
                'max_accounts': max(w['unique_accounts'] for w in tx_metrics),
                'avg_accounts': np.mean([w['unique_accounts'] for w in tx_metrics]),
                'max_concentration': max(w['interaction_concentration'] for w in account_metrics),
                'avg_concentration': np.mean([w['interaction_concentration'] for w in account_metrics])
            }
        else:
            enhanced_features = {
                'max_tx_density': 0,
                'avg_tx_density': 0,
                'peak_volume': 0,
                'volume_volatility': 0,
                'max_accounts': 0,
                'avg_accounts': 0,
                'max_concentration': 0,
                'avg_concentration': 0
            }
            
        return {**price_features, **enhanced_features}

def classify_tokens(input_folder='test', output_folder='test', confidence_threshold=0.6):
    """Classify tokens and handle unmatched files"""
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return
    
    classifier = TokenClassifier()
    
    # Create output directories
    test_yes = os.path.join(output_folder, 'yes')
    test_no = os.path.join(output_folder, 'no')
    later_folder = os.path.join(output_folder, 'later')
    os.makedirs(test_yes, exist_ok=True)
    os.makedirs(test_no, exist_ok=True)
    os.makedirs(later_folder, exist_ok=True)
    
    # Find pairs and unmatched files
    file_pairs, unmatched_files = classifier.find_and_sort_files(input_folder)
    
    # Move unmatched files to 'later' folder
    for unmatched_file in unmatched_files:
        src_path = os.path.join(input_folder, unmatched_file)
        dst_path = os.path.join(later_folder, unmatched_file)
        shutil.copy2(src_path, dst_path)
        print(f"Moved unmatched file to later: {unmatched_file}")
    
    if not file_pairs:
        print("\nNo complete pairs to classify!")
        return
        
    print(f"\nProcessing {len(file_pairs)} complete pairs...")
    
    results = []
    for price_file, enhanced_file in file_pairs:
        try:
            # Extract features
            features = classifier.extract_features(price_file, enhanced_file)
            features_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = predict_model(classifier.model, data=features_df)
            is_pump = bool(prediction['prediction_label'].iloc[0])
            confidence = float(prediction['prediction_score'].iloc[0])
            
            # Determine category based on confidence
            high_confidence_pump = is_pump and confidence >= confidence_threshold
            
            # Copy files to appropriate directory
            target_dir = test_yes if high_confidence_pump else test_no
            token = os.path.basename(price_file).split('_')[1][:8]
            
            shutil.copy2(price_file, os.path.join(target_dir, os.path.basename(price_file)))
            shutil.copy2(enhanced_file, os.path.join(target_dir, os.path.basename(enhanced_file)))
            
            # Store result
            results.append({
                'token': token,
                'prediction': 'pump' if high_confidence_pump else 'no_pump',
                'confidence': confidence
            })
            
            # Print prediction
            status = "✅ PUMP" if high_confidence_pump else "❌ NO PUMP"
            print(f"{status} - Token {token}: {confidence:.1%} confidence")
            
        except Exception as e:
            print(f"Error processing files: {str(e)}")
    
    if results:
        # Save classification results
        results_file = os.path.join(output_folder, 'classification_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        pump_count = sum(1 for r in results if r['prediction'] == 'pump')
        print(f"\nClassification Summary:")
        print(f"Total pairs processed: {len(results)}")
        print(f"Classified as pump: {pump_count}")
        print(f"Classified as no-pump: {len(results) - pump_count}")
        print(f"Unmatched files moved to 'later': {len(unmatched_files)}")
        
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        percentiles = [0, 25, 50, 75, 100]
        print("\nConfidence Distribution:")
        for p in percentiles:
            value = np.percentile(confidences, p)
            print(f"{p}th percentile: {value:.2%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify tokens using trained pump detector')
    parser.add_argument('--input', default='test', help='Input folder containing token files')
    parser.add_argument('--output', default='test', help='Output folder for classification')
    parser.add_argument('--threshold', type=float, default=0.78, help='Confidence threshold for pump classification')
    
    args = parser.parse_args()
    
    print(f"Classifying tokens from {args.input}")
    print(f"Using confidence threshold: {args.threshold:.1%}")
    
    classify_tokens(args.input, args.output, args.threshold)