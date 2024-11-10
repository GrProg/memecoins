# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.

import requests
from datetime import datetime
import json
from typing import List, Optional
import time
import sys
import os

HELIUS_API_KEY = "30b8e7e2-9206-41ca-a392-112845774aef"

def get_all_token_transactions(token_address: str, max_transactions: int = 950) -> Optional[List[dict]]:
    """
    Fetches transactions for a token up to a maximum limit.
    
    Args:
        token_address: The Solana token address to fetch transactions for
        max_transactions: Maximum number of transactions to fetch (default: 950)
    
    Returns:
        List of transactions or None if an error occurs
    """
    base_url = f"https://api.helius.xyz/v0/addresses/{token_address}/transactions"
    all_transactions = []
    before_signature = None
    max_retries = 10
    current_retry = 0
    
    while True:
        try:
            # Check if we've reached the transaction limit
            if len(all_transactions) >= max_transactions:
                print(f"Reached maximum transaction limit of {max_transactions}")
                break

            # Construct URL with pagination parameters
            url = f"{base_url}?api-key={HELIUS_API_KEY}"
            if before_signature:
                url += f"&before={before_signature}"
            
            # Make API request with increased timeout
            response = requests.get(url, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                if current_retry < max_retries:
                    wait_time = 60 * (2 ** current_retry)  # Exponential backoff
                    print(f"Rate limit hit, waiting {wait_time} seconds (retry {current_retry + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    current_retry += 1
                    continue
                else:
                    print("Max retries reached on rate limit")
                    break
            
            # Reset retry counter on successful request
            current_retry = 0
            
            # Handle other errors
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                if all_transactions:  # If we have data, save what we have
                    break
                return None
            
            # Parse response
            transactions = response.json()
            
            # Break if no more transactions
            if not transactions or len(transactions) == 0:
                break
                
            # Add transactions up to the limit
            remaining_slots = max_transactions - len(all_transactions)
            transactions_to_add = transactions[:remaining_slots]
            all_transactions.extend(transactions_to_add)
            
            # Break if we've reached the limit
            if len(all_transactions) >= max_transactions:
                print(f"Reached maximum transaction limit of {max_transactions}")
                break
            
            # Get the signature of the last transaction for pagination
            last_transaction = transactions[-1]
            before_signature = last_transaction.get('signature')
            
            print(f"Fetched {len(transactions_to_add)} transactions. Total so far: {len(all_transactions)}")
            
            # Add delay between requests to avoid rate limiting
            time.sleep(1.5)
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            if "429" in str(e):
                if current_retry < max_retries:
                    wait_time = 60 * (2 ** current_retry)
                    print(f"Rate limit hit, waiting {wait_time} seconds (retry {current_retry + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    current_retry += 1
                    continue
                else:
                    print("Max retries reached on rate limit")
                    break
            elif all_transactions:  # If we have some data, save what we have
                break
            else:
                return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if all_transactions:  # If we have some data, save what we have
                break
            return None
    
    if not all_transactions:
        print("No transactions found")
        return None
        
    # Save all transactions to file
    try:
        # Create 'all' folder if it doesn't exist
        os.makedirs('all', exist_ok=True)
        
        # New filename format
        filename = f"all/transactions_{token_address}.json"
        
        with open(filename, 'w') as f:
            json.dump(all_transactions, f, indent=2)
        print(f"All transactions saved to {filename}")
        
        return all_transactions
    
    except Exception as e:
        print(f"Error saving transactions: {str(e)}")
        return None

# Usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        token_address = sys.argv[1]
    else:
        token_address = input("Enter token address: ").strip()
    
    transactions = get_all_token_transactions(token_address)
    if transactions:
        print(f"Successfully fetched {len(transactions)} total transactions")