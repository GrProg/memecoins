#USE ONLY FOR LIVE DATA
import json
import os
from datetime import datetime
import glob

def clean_transaction_data(input_file, 
                         min_market_cap=4999,  # Minimum market cap threshold
                         max_market_cap=69000,  # Maximum market cap threshold
                         min_sol_amount=0.1,  # Minimum SOL per transaction
                         initial_seconds=10):  # Time window for initial transactions
    
    # Read the JSON file
    with open(input_file, 'r') as f:
        transactions = json.load(f)
    
    # Convert to list if it's not already
    if not isinstance(transactions, list):
        transactions = [transactions]
    
    # Sort transactions by timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    
    # Get the timestamp of the first transaction
    initial_timestamp = transactions[0]['timestamp']
    
    cleaned_transactions = []
    
    for tx in transactions:
        # Skip transactions with too little SOL
        if tx['sol_amount'] < min_sol_amount:
            continue
            
        # Skip transactions with market cap outside thresholds
        if not (min_market_cap <= tx['market_cap'] <= max_market_cap):
            continue
            
        # If transaction is within initial_seconds of first transaction,
        # set market cap to 5000
        if tx['timestamp'] - initial_timestamp <= initial_seconds:
            tx['market_cap'] = 5000
            tx['price_in_usd'] = 5000 / tx['token_amount']
            
        cleaned_transactions.append(tx)
    
    return cleaned_transactions

def process_directory():
    # Get all price_history*.json files from the 'yes' directory
    input_files = glob.glob('yes/price_history*.json')
    
    for input_file in input_files:
        try:
            # Clean the data
            cleaned_data = clean_transaction_data(input_file)
            
            # Create output filename
            filename = os.path.basename(input_file)
            output_file = f'yes/{filename}'
            
            # Save cleaned data
            with open(output_file, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
                
            print(f"Processed {input_file} -> {output_file}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    # Create 'yes' directory if it doesn't exist
    os.makedirs('yes', exist_ok=True)
    process_directory()