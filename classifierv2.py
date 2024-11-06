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
from typing import List, Dict, Any
import json
import os
import shutil

class PumpDumpClassifier:
    def __init__(self,
                 volatility_threshold: float = 0.4,    # Allow high initial volatility
                 min_trend_duration: int = 20,         # Minimum seconds for trend phase
                 dump_threshold: float = 60,           # Minimum % drop for dump
                 max_market_cap: float = 68000):       # Maximum market cap to consider
        
        self.volatility_threshold = volatility_threshold
        self.min_trend_duration = min_trend_duration
        self.dump_threshold = dump_threshold
        self.max_market_cap = max_market_cap

    def is_pump_and_dump(self, trades: List[Dict[str, Any]]) -> bool:
        """
        Determine if the trading pattern represents a pump and dump scheme.
        Returns True if pattern matches, False otherwise.
        """
        # Prepare and filter data
        df = pd.DataFrame(trades)
        df = df[df['market_cap'] <= self.max_market_cap]
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        
        if len(df) < 10:  # Need minimum data points
            return False
            
        # Calculate key metrics
        df['price_pct_change'] = df['price_in_usd'].pct_change()
        df['volatility'] = df['price_pct_change'].rolling(5).std()
        
        # Find initial spike phase
        initial_section = df.iloc[:min(20, len(df))]
        has_initial_volatility = (initial_section['volatility'].max() > self.volatility_threshold and 
                                initial_section['price_in_usd'].max() > initial_section['price_in_usd'].iloc[0] * 1.2)
        
        if not has_initial_volatility:
            return False
            
        # Analyze trend phase
        middle_section = df.iloc[20:int(len(df)*0.8)]
        if len(middle_section) < self.min_trend_duration:
            return False
            
        # Calculate overall trend
        trend_start_price = middle_section['price_in_usd'].iloc[0]
        trend_end_price = middle_section['price_in_usd'].iloc[-1]
        trend_duration = middle_section['timestamp'].iloc[-1] - middle_section['timestamp'].iloc[0]
        
        has_trend = (trend_end_price > trend_start_price and 
                    trend_duration >= self.min_trend_duration)
        
        if not has_trend:
            return False
            
        # Check for dump
        end_section = df.iloc[-10:]
        peak_price = df['price_in_usd'].max()
        final_price = end_section['price_in_usd'].iloc[-1]
        dump_percentage = ((peak_price - final_price) / peak_price) * 100
        
        has_dump = dump_percentage >= self.dump_threshold
        
        # Return True only if we see all three phases
        return has_initial_volatility and has_trend and has_dump

def analyze_trading_data(trades_data: List[Dict[str, Any]]) -> bool:
    classifier = PumpDumpClassifier()
    return classifier.is_pump_and_dump(trades_data)

def process_all_files():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create 'yes' directory in the script's directory if it doesn't exist
    yes_dir = os.path.join(script_dir, 'yes')
    if not os.path.exists(yes_dir):
        os.makedirs(yes_dir)
    
    # Path to 'all' directory
    all_dir = os.path.join(script_dir, 'all')
    
    # Process all JSON files in the 'all' directory
    files_processed = 0
    pump_detected = 0
    
    for filename in os.listdir(all_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(all_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    trades_data = json.load(f)
                
                is_pump = analyze_trading_data(trades_data)
                
                if is_pump:
                    # Copy file to 'yes' directory
                    shutil.copy2(file_path, os.path.join(yes_dir, filename))
                    pump_detected += 1
                
                files_processed += 1
                print(f"Processed {filename}: {'Pump detected' if is_pump else 'No pump detected'}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing complete:")
    print(f"Total files processed: {files_processed}")
    print(f"Pump and dump patterns detected: {pump_detected}")

if __name__ == "__main__":
    process_all_files()

