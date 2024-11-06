import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Tuple, Dict
import re
from datetime import datetime

def find_matching_files(directory: str) -> List[Tuple[str, str]]:
    """Find matching enhanced and price history files"""
    files = os.listdir(directory)
    pairs = []
    
    enhanced_files = {}
    price_files = {}
    
    for file in files:
        if not file.endswith('.json'):
            continue
            
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
    
    for key in enhanced_files:
        if key in price_files:
            pairs.append((
                os.path.join(directory, price_files[key]),
                os.path.join(directory, enhanced_files[key])
            ))
    
    return pairs

def analyze_token_data(price_file: str, enhanced_file: str) -> Dict:
    """Analyze transaction and market cap data for a single token"""
    try:
        # Load data
        with open(price_file, 'r') as f:
            price_data = pd.DataFrame(json.load(f))
        
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        
        # Filter out market caps above 69k
        price_data = price_data[price_data['market_cap'] <= 69000]
        
        if len(price_data) == 0:
            return None
            
        # Get max market cap
        max_market_cap = price_data['market_cap'].max()
        initial_market_cap = price_data['market_cap'].iloc[0]
        final_market_cap = price_data['market_cap'].iloc[-1]
        
        # Calculate total transactions
        total_tx = sum(entry['window_metrics']['transaction_metrics']['total_transactions'] 
                      for entry in enhanced_data)
        
        # Calculate dump percentage
        dump_percentage = ((max_market_cap - final_market_cap) / max_market_cap) * 100 if max_market_cap > 0 else 0
        
        return {
            'token': os.path.basename(price_file)[:8],
            'max_market_cap': max_market_cap,
            'dump_percentage': dump_percentage,
            'total_transactions': total_tx,
            'initial_market_cap': initial_market_cap,
            'final_market_cap': final_market_cap
        }
    except Exception as e:
        print(f"Error processing {price_file}: {str(e)}")
        return None

def create_visualization(results: List[Dict]):
    """Create comprehensive visualization of the analysis"""
    # Filter out None values and create DataFrame
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No valid results to analyze")
        return
        
    df = pd.DataFrame(valid_results)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Transactions vs Market Cap
    ax1.scatter(df['total_transactions'], df['max_market_cap'])
    ax1.set_xlabel('Total Transactions')
    ax1.set_ylabel('Max Market Cap (USD)')
    ax1.set_title('Transactions vs Max Market Cap')
    
    # Add trend line if there are enough points
    if len(df) > 1:
        z = np.polyfit(df['total_transactions'], df['max_market_cap'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['total_transactions'].min(), df['total_transactions'].max(), 100)
        ax1.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    # 2. Dump Percentage Distribution
    sns.histplot(data=df, x='dump_percentage', bins=20, ax=ax2)
    ax2.set_title('Distribution of Dump Percentages')
    ax2.set_xlabel('Dump Percentage')
    
    # 3. Initial vs Final Market Cap
    ax3.scatter(df['initial_market_cap'], df['final_market_cap'])
    ax3.set_xlabel('Initial Market Cap (USD)')
    ax3.set_ylabel('Final Market Cap (USD)')
    ax3.set_title('Initial vs Final Market Cap')
    
    # 4. Transaction Distribution
    sns.histplot(data=df, x='total_transactions', bins=20, ax=ax4)
    ax4.set_title('Distribution of Transaction Counts')
    ax4.set_xlabel('Number of Transactions')
    
    # Add overall statistics
    stats_text = (
        f"Total Tokens Analyzed: {len(df)}\n"
        f"Avg Max Market Cap: ${df['max_market_cap'].mean():.2f}\n"
        f"Avg Dump Percentage: {df['dump_percentage'].mean():.2f}%\n"
        f"Avg Transactions: {df['total_transactions'].mean():.2f}\n"
        f"Correlation (Tx vs MCap): {df['total_transactions'].corr(df['max_market_cap']):.2f}"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('token_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total tokens analyzed: {len(df)}")
    print(f"Average max market cap: ${df['max_market_cap'].mean():.2f}")
    print(f"Average dump percentage: {df['dump_percentage'].mean():.2f}%")
    print(f"Average transactions: {df['total_transactions'].mean():.2f}")
    if len(df) > 1:
        print(f"Correlation (Tx vs MCap): {df['total_transactions'].corr(df['max_market_cap']):.2f}")
    
    # Save detailed results
    df.to_csv('token_analysis_results.csv', index=False)

def main():
    # Find and analyze all token pairs
    pairs = find_matching_files('yes')
    print(f"Found {len(pairs)} token pairs to analyze")
    
    results = []
    for price_file, enhanced_file in pairs:
        result = analyze_token_data(price_file, enhanced_file)
        if result:
            results.append(result)
    
    # Create visualization and analysis
    create_visualization(results)

if __name__ == "__main__":
    main()