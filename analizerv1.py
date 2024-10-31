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
from datetime import datetime
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import os
import re
def find_matching_files(directory: str) -> List[Tuple[str, str]]:
        """Find matching enhanced and price history files for the same token"""
        files = os.listdir(directory)
        pairs = []

        # Create dictionaries to store files by their token and timestamp
        enhanced_files = {}
        price_files = {}

        for file in files:
            if not file.endswith('.json'):
                continue

            # Extract token address and timestamp
            # Pattern for both enhanced and price history files
            if file.startswith('enhanced_'):
                pattern = r'enhanced_(.+)_(\d+_\d+).json'
            else:
                pattern = r'price_history_(.+)_(\d+_\d+).json'

            match = re.match(pattern, file)
            if match:
                token_address = match.group(1)
                timestamp = match.group(2)
                key = f"{token_address}_{timestamp}"

                if file.startswith('enhanced_'):
                    enhanced_files[key] = file
                else:
                    price_files[key] = file

        # Match the files
        for key in enhanced_files.keys():
            if key in price_files:
                enhanced_file = os.path.join(directory, enhanced_files[key])
                price_file = os.path.join(directory, price_files[key])
                pairs.append((price_file, enhanced_file))
                print(f"Found matching pair for token {key.split('_')[0][:8]}...")

        return pairs
class PumpDumpAnalyzer:
    def __init__(self):
        self.simple_data = None
        self.enhanced_data = None
        self.token_address = None
        self.timestamp = None
        
    @staticmethod
    def find_matching_files(directory: str) -> List[Tuple[str, str]]:
        return find_matching_files(directory)
        

    def load_data(self, price_history_path: str, enhanced_path: str):
        """Load both price history and enhanced JSON data"""
        with open(price_history_path, 'r') as f:
            price_data = json.load(f)
            self.simple_data = pd.DataFrame(price_data)
            
        with open(enhanced_path, 'r') as f:
            enhanced_data = json.load(f)
            self.enhanced_data = pd.DataFrame(enhanced_data)
        
        # Extract token address from filename
        pattern = r'_(\w{44})_'
        match = re.search(pattern, price_history_path)
        if match:
            self.token_address = match.group(1)
            
        # Convert timestamps
        self.simple_data['datetime'] = pd.to_datetime(self.simple_data['timestamp'], unit='s')
        self.enhanced_data['datetime'] = pd.to_datetime(self.enhanced_data['timestamp'], unit='s')

    def analyze_token(self) -> Dict:
        """Comprehensive analysis of a single token"""
        analysis = {
            'token_address': self.token_address,
            'price_metrics': self._analyze_price_metrics(),
            'volume_metrics': self._analyze_volume_metrics(),
            'transaction_metrics': self._analyze_transaction_metrics(),
            'account_metrics': self._analyze_account_metrics()
        }
        return analysis
        
    def _analyze_price_metrics(self) -> Dict:
        """Analyze price and market cap patterns"""
        metrics = {
            'max_market_cap': self.simple_data['market_cap'].max(),
            'min_market_cap': self.simple_data['market_cap'].min(),
            'final_market_cap': self.simple_data['market_cap'].iloc[-1],
            'time_to_peak_minutes': None,
            'price_volatility': self.simple_data['price_in_usd'].std() / self.simple_data['price_in_usd'].mean(),
            'dump_percentage': None,
            'pump_duration_minutes': None
        }
        
        # Calculate time to peak
        peak_idx = self.simple_data['market_cap'].idxmax()
        start_time = self.simple_data['datetime'].min()
        peak_time = self.simple_data.loc[peak_idx, 'datetime']
        metrics['time_to_peak_minutes'] = (peak_time - start_time).total_seconds() / 60
        
        # Calculate dump percentage
        peak_mcap = metrics['max_market_cap']
        end_mcap = metrics['final_market_cap']
        metrics['dump_percentage'] = ((peak_mcap - end_mcap) / peak_mcap) * 100
        
        # Calculate pump duration
        metrics['pump_duration_minutes'] = (self.simple_data['datetime'].max() - 
                                         self.simple_data['datetime'].min()).total_seconds() / 60
        
        return metrics
    
    def _analyze_volume_metrics(self) -> Dict:
        """Analyze trading volume patterns"""
        metrics = {
            'max_tx_density': 0,
            'avg_tx_density': 0,
            'peak_volume': 0,
            'volume_volatility': 0
        }
        
        # Extract metrics from enhanced data
        densities = []
        volumes = []
        volatilities = []
        
        for _, row in self.enhanced_data.iterrows():
            window = row['window_metrics']
            densities.append(window['transaction_metrics']['tx_density'])
            volumes.append(window['sol_metrics']['total_sol_volume'])
            volatilities.append(window['sol_metrics']['sol_flow_volatility'])
        
        metrics['max_tx_density'] = max(densities) if densities else 0
        metrics['avg_tx_density'] = np.mean(densities) if densities else 0
        metrics['peak_volume'] = max(volumes) if volumes else 0
        metrics['volume_volatility'] = np.mean(volatilities) if volatilities else 0
        
        return metrics
    
    def _analyze_transaction_metrics(self) -> Dict:
        """Analyze transaction patterns"""
        metrics = {
            'total_transactions': 0,
            'unique_accounts': 0,
            'avg_tx_per_block': 0,
            'max_tx_per_block': 0
        }
        
        # Extract transaction metrics
        tx_counts = []
        account_counts = []
        
        for _, row in self.enhanced_data.iterrows():
            window = row['window_metrics']
            tx_counts.append(window['transaction_metrics']['total_transactions'])
            account_counts.append(window['transaction_metrics']['unique_accounts'])
        
        metrics['total_transactions'] = sum(tx_counts)
        metrics['unique_accounts'] = max(account_counts) if account_counts else 0
        metrics['avg_tx_per_block'] = np.mean(tx_counts) if tx_counts else 0
        metrics['max_tx_per_block'] = max(tx_counts) if tx_counts else 0
        
        return metrics
    
    def _analyze_account_metrics(self) -> Dict:
        """Analyze account behavior patterns"""
        metrics = {
            'max_account_concentration': 0,
            'avg_account_concentration': 0,
            'peak_unique_accounts': 0,
            'avg_unique_accounts': 0
        }
        
        concentrations = []
        unique_accounts = []
        
        for _, row in self.enhanced_data.iterrows():
            window = row['window_metrics']
            concentrations.append(window['account_metrics']['interaction_concentration'])
            unique_accounts.append(window['transaction_metrics']['unique_accounts'])
        
        metrics['max_account_concentration'] = max(concentrations) if concentrations else 0
        metrics['avg_account_concentration'] = np.mean(concentrations) if concentrations else 0
        metrics['peak_unique_accounts'] = max(unique_accounts) if unique_accounts else 0
        metrics['avg_unique_accounts'] = np.mean(unique_accounts) if unique_accounts else 0
        
        return metrics
    
    def plot_analysis(self, save_path: str = None):
        """Create visualization of key metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Market Cap Evolution
        self.simple_data.plot(x='datetime', y='market_cap', ax=ax1)
        ax1.set_title(f'Market Cap Evolution - {self.token_address[:8]}...')
        ax1.set_ylabel('Market Cap (USD)')
        
        # Transaction Density
        densities = []
        times = []
        for idx, row in self.enhanced_data.iterrows():
            densities.append(row['window_metrics']['transaction_metrics']['tx_density'])
            times.append(row['datetime'])
        ax2.plot(times, densities)
        ax2.set_title('Transaction Density')
        
        # Account Activity
        accounts = []
        for idx, row in self.enhanced_data.iterrows():
            accounts.append(row['window_metrics']['transaction_metrics']['unique_accounts'])
        ax3.plot(times, accounts)
        ax3.set_title('Unique Accounts')
        
        # Volume Profile
        volumes = []
        for idx, row in self.enhanced_data.iterrows():
            volumes.append(row['window_metrics']['sol_metrics']['total_sol_volume'])
        ax4.plot(times, volumes)
        ax4.set_title('Trading Volume (SOL)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def analyze_all_tokens(directory: str = 'yes') -> List[Dict]:
    """Analyze all token pairs in the directory"""
    pairs = find_matching_files(directory)
    results = []
    
    print(f"Found {len(pairs)} token pairs to analyze")
    
    for price_file, enhanced_file in pairs:
        try:
            print(f"\nAnalyzing files:")
            print(f"Price history: {os.path.basename(price_file)}")
            print(f"Enhanced data: {os.path.basename(enhanced_file)}")
            
            analyzer = PumpDumpAnalyzer()
            analyzer.load_data(price_file, enhanced_file)
            
            # Run analysis
            analysis = analyzer.analyze_token()
            results.append(analysis)
            
            # Generate plots
            plot_file = f"analysis_{analyzer.token_address[:8]}.png"
            analyzer.plot_analysis(plot_file)
            
            # Print summary
            price_metrics = analysis['price_metrics']
            print(f"Max Market Cap: ${price_metrics['max_market_cap']:.2f}")
            print(f"Time to Peak: {price_metrics['time_to_peak_minutes']:.2f} minutes")
            print(f"Dump Percentage: {price_metrics['dump_percentage']:.2f}%")
            
        except Exception as e:
            print(f"Error analyzing files:\n{str(e)}")
            traceback.print_exc()
            continue
    

if __name__ == "__main__":
    import traceback
    
    print("Starting analysis of all tokens...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in 'yes' directory: {os.listdir('yes')}")
    
    try:
        results = analyze_all_tokens()
        
        if results:
            # Save results
            with open('aggregate_analysis.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print aggregate statistics
            print("\nAggregate Statistics:")
            max_mcaps = [r['price_metrics']['max_market_cap'] for r in results]
            dump_pcts = [r['price_metrics']['dump_percentage'] for r in results]
            times_to_peak = [r['price_metrics']['time_to_peak_minutes'] for r in results]
            
            print(f"Average Max Market Cap: ${np.mean(max_mcaps):.2f}")
            print(f"Average Dump Percentage: {np.mean(dump_pcts):.2f}%")
            print(f"Average Time to Peak: {np.mean(times_to_peak):.2f} minutes")
            print(f"\nTotal tokens analyzed: {len(results)}")
        else:
            print("No results generated. Check if files are in the correct location and format.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
