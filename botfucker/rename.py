import os
import json
import re
from typing import List, Dict, Tuple

class FileRenamer:
    def __init__(self, file_path: str):
        """
        Initialize the renamer with a file path.
        Args:
            file_path (str): Path to the JSON file to process
        """
        self.file_path = file_path
        self.dir_path = os.path.dirname(file_path)
        self.filename = os.path.basename(file_path)

    def _load_transactions(self) -> List[Dict]:
        """Load transactions from the JSON file."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading file {self.file_path}: {str(e)}")
            return []

    def _analyze_transactions(self, transactions: List[Dict]) -> Tuple[int, int, float]:
        """
        Analyze transactions for counts and special amounts.
        Args:
            transactions (List[Dict]): List of transaction data
        Returns:
            Tuple[int, int, float]: (total_transactions, special_transactions, sol_amount)
        """
        total_tx = len(transactions)
        
        # Track special transactions and sum their amounts
        special_tx = 0
        total_special_amount = 0.0
        
        for tx in transactions:
            sol_amount = tx.get('sol_amount', 0)
            if not f"{sol_amount:.8f}".startswith('0.01') and sol_amount > 0.015:
                special_tx += 1
                total_special_amount += sol_amount
                
        return total_tx, special_tx, total_special_amount

    def _generate_new_filename(self, total_tx: int, special_tx: int, sol_amount: float) -> str:
        """
        Generate new filename with all components.
        Args:
            total_tx (int): Total number of transactions
            special_tx (int): Number of special transactions
            sol_amount (float): Total SOL amount from special transactions
        Returns:
            str: New filename
        """
        # Extract components from current filename
        match = re.match(r'price_history_(.+)_(\d+)', self.filename)
        if not match:
            return self.filename

        token_address = match.group(1)
        mcap = match.group(2).split('.')[0]  # Remove any extension
        
        # Format the SOL amount to 2 decimal places
        sol_amount_str = f"{sol_amount:.2f}"
        
        # Format the new filename with all components
        return f"price_history_{token_address}_{mcap}_{sol_amount_str}_{total_tx}_{special_tx}.json"

    def rename_file(self) -> bool:
        """
        Process file and rename it with all components.
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load and process transactions
            transactions = self._load_transactions()
            if not transactions:
                print("No transactions found in file")
                return False

            # Get transaction counts and SOL amount
            total_tx, special_tx, sol_amount = self._analyze_transactions(transactions)

            # Generate new filename
            new_filename = self._generate_new_filename(total_tx, special_tx, sol_amount)
            new_path = os.path.join(self.dir_path, new_filename)

            # Rename file
            os.rename(self.file_path, new_path)
            print(f"Successfully renamed file to: {new_filename}")
            print(f"Total SOL amount from special transactions: {sol_amount:.2f}")
            print(f"Total transactions: {total_tx}")
            print(f"Special transactions (sol_amount not starting with 0.01): {special_tx}")
            return True

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return False

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python rename.py <price_history_file.json>")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return

    renamer = FileRenamer(file_path)
    renamer.rename_file()

if __name__ == "__main__":
    main()