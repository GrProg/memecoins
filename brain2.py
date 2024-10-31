# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.
import pandas as pd
import numpy as np
from pycaret.classification import *
import json
import os
from pathlib import Path
import shutil


def sort_test_predictions(test_folder='test', relative_threshold=0.6):
    """
    Sort test files into yes/no folders based on relative confidence scores
    relative_threshold: how close to peak confidence (0-1) to consider as positive
    """
    # Load the predictions
    predictions_df = pd.read_csv('test_predictions.csv')
    
    # Calculate confidence range
    max_confidence = predictions_df['prediction_score'].max()
    min_confidence = predictions_df['prediction_score'].min()
    confidence_range = max_confidence - min_confidence
    
    # Calculate dynamic threshold based on range
    dynamic_threshold = max_confidence - (confidence_range * relative_threshold)
    
    print(f"\nConfidence Statistics:")
    print(f"Peak confidence: {max_confidence:.2%}")
    print(f"Lowest confidence: {min_confidence:.2%}")
    print(f"Dynamic threshold: {dynamic_threshold:.2%}")
    
    # Create yes/no folders inside test folder
    test_yes_folder = os.path.join(test_folder, 'yes')
    test_no_folder = os.path.join(test_folder, 'no')
    
    os.makedirs(test_yes_folder, exist_ok=True)
    os.makedirs(test_no_folder, exist_ok=True)
    
    # Get all JSON files in test folder
    test_files = list(Path(test_folder).glob('*.json'))
    
    # Process each test file
    print("\nSorting test files...")
    files_processed = 0
    confidence_scores = []
    
    for file_path in test_files:
        try:
            # Load the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get the first transaction's signature prefix
            if data and len(data) > 0:
                first_sig = data[0]['signature'][:8]
                
                # Find predictions for this signature
                file_predictions = predictions_df[
                    predictions_df['signature'].str.startswith(first_sig)
                ]
                
                if not file_predictions.empty:
                    # Calculate average confidence for this file
                    avg_confidence = file_predictions['prediction_score'].mean()
                    confidence_scores.append({
                        'file': file_path,
                        'confidence': avg_confidence
                    })
                
                files_processed += 1
                if files_processed % 10 == 0:
                    print(f"Processed {files_processed} files...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Sort files by confidence
    confidence_scores.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Process sorted files
    for entry in confidence_scores:
        file_path = entry['file']
        confidence = entry['confidence']
        
        # Determine if it's closer to peak confidence
        is_pump = confidence >= dynamic_threshold
        
        # Copy file to appropriate folder
        target_folder = test_yes_folder if is_pump else test_no_folder
        shutil.copy2(file_path, target_folder)
        
        # Calculate how close to peak/bottom
        if is_pump:
            proximity_to_peak = (confidence - dynamic_threshold) / (max_confidence - dynamic_threshold)
            print(f"✅ {file_path.name} -> pump ({confidence:.2%}, {proximity_to_peak:.2%} close to peak)")
        else:
            proximity_to_bottom = (confidence - min_confidence) / (dynamic_threshold - min_confidence)
            print(f"❌ {file_path.name} -> not pump ({confidence:.2%}, {proximity_to_bottom:.2%} close to bottom)")
    
    # Print summary with distribution analysis
    yes_count = len(list(Path(test_yes_folder).glob('*.json')))
    no_count = len(list(Path(test_no_folder).glob('*.json')))
    
    print(f"\nProcessing complete!")
    print(f"Files processed: {files_processed}")
    print(f"Files marked as pump: {yes_count} ({yes_count/files_processed:.1%})")
    print(f"Files marked as non-pump: {no_count} ({no_count/files_processed:.1%})")
    
    # Distribution analysis
    print("\nConfidence Distribution:")
    confidences = [score['confidence'] for score in confidence_scores]
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        value = np.percentile(confidences, p)
        print(f"{p}th percentile: {value:.2%}")

if __name__ == "__main__":
    try:
        # Load the trained model
        print("Loading trained model...")
        model = load_model('pump_detector_final')
        
        # Sort the predictions with relative threshold
        # Adjust this value to change the split point relative to peak confidence
        sort_test_predictions(relative_threshold=0.01)
        
    except Exception as e:
        print(f"Error: {str(e)}")