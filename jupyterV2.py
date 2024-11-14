import requests
from datetime import datetime
import json
from typing import List, Optional
import time
import sys
import os

HELIUS_API_KEY = "30b8e7e2-9206-41ca-a392-112845774aef"

class TransactionFetcher:
    def __init__(self, token_address: str, max_transactions: int =  50050):
        self.token_address = token_address
        self.max_transactions = max_transactions
        self.base_url = f"https://api.helius.xyz/v0/addresses/{token_address}/transactions"
        self.max_retries = 10
        
    def get_existing_file_path(self) -> str:
        """Get path to existing transaction file"""
        return os.path.join('all', f'transactions_{self.token_address}.json')
        
    def get_latest_transaction(self) -> Optional[dict]:
        """Get the most recent transaction from existing file"""
        file_path = self.get_existing_file_path()
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                transactions = json.load(f)
                if transactions:
                    # Sort by timestamp to ensure we get the latest
                    transactions.sort(key=lambda x: x['timestamp'], reverse=True)
                    latest_tx = transactions[0]
                    print(f"Latest transaction timestamp: {datetime.fromtimestamp(latest_tx['timestamp'])}")
                    print(f"Latest signature: {latest_tx['signature'][:32]}...")
                    return latest_tx
                return None
        except Exception as e:
            print(f"Error reading existing transactions: {str(e)}")
            return None

    def fetch_new_transactions(self) -> List[dict]:
        """Fetch new transactions with early stopping when we hit known transactions"""
        all_transactions = []
        current_retry = 0
        latest_tx = self.get_latest_transaction()

        if not latest_tx:
            print("No existing transactions found - fetching all transactions")
        else:
            print("Fetching only new transactions...")

        while True:
            try:
                # Construct basic URL
                url = f"{self.base_url}?api-key={HELIUS_API_KEY}"
                
                # Add before parameter if we have transactions
                if all_transactions:
                    url += f"&before={all_transactions[-1]['signature']}"
                
                # Make API request
                response = requests.get(url, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    if current_retry < self.max_retries:
                        wait_time = 60 * (2 ** current_retry)
                        print(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        current_retry += 1
                        continue
                    break
                
                current_retry = 0
                
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    break
                
                # Parse response
                batch = response.json()
                if not batch:
                    break
                    
                # If we have a latest transaction, check if we've hit it
                if latest_tx:
                    for i, tx in enumerate(batch):
                        if tx['signature'] == latest_tx['signature']:
                            # We've hit our latest known transaction
                            if i > 0:
                                all_transactions.extend(batch[:i])
                                print(f"Found existing transaction, stopping fetch.")
                            return all_transactions
                
                # Add batch to our transactions
                all_transactions.extend(batch)
                print(f"Fetched {len(batch)} new transactions. Total: {len(all_transactions)}")
                
                # Check limits
                if len(all_transactions) >= self.max_transactions:
                    print(f"Reached maximum transaction limit of {self.max_transactions}")
                    break
                
                # Add delay to avoid rate limiting
                time.sleep(1.5)
                
            except Exception as e:
                print(f"Error fetching transactions: {str(e)}")
                break
                
        return all_transactions

    def update_transaction_file(self) -> bool:
        """Update transaction file with new transactions"""
        try:
            os.makedirs('all', exist_ok=True)
            file_path = self.get_existing_file_path()
            
            # Fetch new transactions
            new_transactions = self.fetch_new_transactions()
            
            if not new_transactions:
                print("No new transactions found")
                return True
                
            # If file exists, merge transactions
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    existing_transactions = json.load(f)
                    
                # Create signature set for quick lookup
                existing_signatures = {tx.get('signature') for tx in existing_transactions}
                
                # Add only new unique transactions
                added_count = 0
                for tx in new_transactions:
                    if tx.get('signature') not in existing_signatures:
                        existing_transactions.append(tx)
                        added_count += 1
                        
                print(f"Added {added_count} new unique transactions")
                transactions_to_save = existing_transactions
            else:
                print(f"Creating new transaction file with {len(new_transactions)} transactions")
                transactions_to_save = new_transactions
            
            # Sort by timestamp before saving
            transactions_to_save.sort(key=lambda x: x['timestamp'])
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(transactions_to_save, f, indent=2)
                
            print(f"Successfully saved transactions to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error updating transactions: {str(e)}")
            return False

def main():
    if len(sys.argv) > 1:
        token_address = sys.argv[1]
    else:
        token_address = input("Enter token address: ").strip()
    
    fetcher = TransactionFetcher(token_address)
    success = fetcher.update_transaction_file()
    
    if success:
        print("Transaction update completed successfully")
    else:
        print("Error updating transactions")
        sys.exit(1)

if __name__ == "__main__":
    main()