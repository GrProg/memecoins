import os
import json
from datetime import datetime
from typing import List, Dict, Any

class TransactionCleaner:
    def __init__(self, min_sol_amount: float = 0.1):
        self.min_sol_amount = min_sol_amount
        
    def clean_file(self, input_file: str) -> None:
        """Clean a single JSON file by removing small transactions"""
        try:
            # Read input file
            with open(input_file, 'r') as f:
                transactions = json.load(f)
                
            if not isinstance(transactions, list):
                print(f"Error: {input_file} does not contain a list of transactions")
                return
                
            original_count = len(transactions)
            
            # Filter transactions
            filtered_transactions = [
                tx for tx in transactions
                if 'sol_amount' in tx and float(tx['sol_amount']) >= self.min_sol_amount
            ]
            
            removed_count = original_count - len(filtered_transactions)
            
            # Create backup of original file
            backup_path = f"{input_file}.bak"
            if not os.path.exists(backup_path):
                os.rename(input_file, backup_path)
            
            # Write filtered data back to original file
            with open(input_file, 'w') as f:
                json.dump(filtered_transactions, f, indent=2)
                
            print(f"Processed {input_file}:")
            print(f"  Original transactions: {original_count}")
            print(f"  Removed transactions: {removed_count}")
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
        print(f"Minimum SOL amount threshold: {self.min_sol_amount}")
        
        # Process each file
        for filename in price_files:
            file_path = os.path.join(directory, filename)
            self.clean_file(file_path)

def clean_transactions(directory: str = 'yes', min_sol: float = 0.1) -> None:
    """Clean all transaction files in the specified directory"""
    cleaner = TransactionCleaner(min_sol_amount=min_sol)
    cleaner.process_directory(directory)

if __name__ == "__main__":
    clean_transactions()