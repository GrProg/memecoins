import os
import json
from datetime import datetime
from typing import List, Dict, Any

class TransactionCleaner:
    def __init__(self, 
                 min_sol_amount: float = 0.001,
                 min_market_cap: float = 1,
                 max_market_cap: float = 64000):  # 10B default max market cap
        self.min_sol_amount = min_sol_amount
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        
    def clean_file(self, input_file: str) -> None:
        """Clean a single JSON file by removing transactions outside specified thresholds"""
        try:
            # Read input file
            with open(input_file, 'r') as f:
                transactions = json.load(f)
                
            if not isinstance(transactions, list):
                print(f"Error: {input_file} does not contain a list of transactions")
                return
                
            original_count = len(transactions)
            
            # Filter transactions based on SOL amount and market cap range
            filtered_transactions = [
                tx for tx in transactions
                if (
                    'sol_amount' in tx and 
                    float(tx['sol_amount']) >= self.min_sol_amount and
                    'market_cap' in tx and 
                    self.min_market_cap <= float(tx['market_cap']) <= self.max_market_cap
                )
            ]
            
            removed_count = original_count - len(filtered_transactions)
            
            # Create backup of original file
            backup_path = f"{input_file}.bak"
            if not os.path.exists(backup_path):
                os.rename(input_file, backup_path)
            
            # Write filtered data back to original file
            with open(input_file, 'w') as f:
                json.dump(filtered_transactions, f, indent=2)
                
            # Calculate filtering statistics
            sol_filtered = len([
                tx for tx in transactions
                if 'sol_amount' in tx and float(tx['sol_amount']) < self.min_sol_amount
            ])
            
            low_market_cap_filtered = len([
                tx for tx in transactions
                if 'market_cap' in tx and float(tx['market_cap']) < self.min_market_cap
            ])
            
            high_market_cap_filtered = len([
                tx for tx in transactions
                if 'market_cap' in tx and float(tx['market_cap']) > self.max_market_cap
            ])
            
            print(f"\nProcessed {input_file}:")
            print(f"  Original transactions: {original_count}")
            print(f"  Removed transactions: {removed_count}")
            print(f"    - Due to low SOL amount: {sol_filtered}")
            print(f"    - Due to low market cap: {low_market_cap_filtered}")
            print(f"    - Due to high market cap: {high_market_cap_filtered}")
            print(f"  Remaining transactions: {len(filtered_transactions)}")
            print(f"  Original file backed up to: {backup_path}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

    def process_directory(self, directory: str) -> None:
        """Process all price history JSON files in a directory"""
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist")
            return
            
        # Find all price history JSON files
        price_files = [
            f for f in os.listdir(directory)
            if f.startswith('price_history_') and f.endswith('.json')
        ]
        
        if not price_files:
            print(f"No price history files found in {directory}")
            return
            
        print(f"Found {len(price_files)} price history files to process")
        print(f"Filtering criteria:")
        print(f"  - Minimum SOL amount: {self.min_sol_amount}")
        print(f"  - Market cap range: {self.min_market_cap:,.2f} - {self.max_market_cap:,.2f}")
        
        # Process each file
        for filename in price_files:
            file_path = os.path.join(directory, filename)
            self.clean_file(file_path)

def clean_transactions(
    directory: str = '.',
    min_sol: float = 0.001,
    min_market_cap: float = 1,
    max_market_cap: float = 64000
) -> None:
    """Clean all transaction files in the specified directory"""
    cleaner = TransactionCleaner(
        min_sol_amount=min_sol,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap
    )
    cleaner.process_directory(directory)

if __name__ == "__main__":
    clean_transactions()