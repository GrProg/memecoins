# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys

def parse_transaction(tx: Dict, sol_price_usd: float = 170, total_supply: float = 1000000) -> Optional[Dict]:
    try:
        # Skip if no token transfers
        if 'tokenTransfers' not in tx or 'accountData' not in tx:
            return None

        # Get timestamp
        timestamp = tx.get('timestamp')
        if not timestamp:
            return None

        # Find token amount from tokenTransfers
        token_amount = None
        to_user = None
        for transfer in tx['tokenTransfers']:
            if 'tokenAmount' in transfer:
                token_amount = float(transfer['tokenAmount'])
                to_user = transfer['toUserAccount']
                break

        if not token_amount or not to_user:
            return None

        # Find SOL amount from accountData
        sol_amount = None
        for account in tx['accountData']:
            if account['account'] == to_user:
                sol_amount = abs(account['nativeBalanceChange']) / 1e9
                break

        if not sol_amount:
            return None

        # Calculate price in SOL and USD
        price_in_sol = sol_amount / token_amount if token_amount != 0 else 0
        price_in_usd = price_in_sol * sol_price_usd
        market_cap = price_in_usd * total_supply
        
        # Filter out transactions with value less than $1
        transaction_value_usd = sol_amount * sol_price_usd
        if transaction_value_usd < 10:
            return None

        return {
            'timestamp': timestamp,
            'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'token_amount': token_amount,
            'sol_amount': sol_amount,
            'price_in_sol': price_in_sol,
            'price_in_usd': price_in_usd,
            'market_cap': market_cap,
            'signature': tx.get('signature', 'unknown')
        }
        
    except Exception as e:
        print(f"Error processing transaction {tx.get('signature', 'unknown')}: {str(e)}")
        return None

def main():
    # Configuration
    sol_price_usd = 177  # Current SOL price in USD
    total_supply = 1000000000  # Token's total supply

    # Get input file from command line argument
    if len(sys.argv) != 2:
        print("Please provide input file path")
        sys.exit(1)

    input_file = sys.argv[1]

    # Load transactions from file
    try:
        with open(input_file, 'r') as f:
            transactions = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        sys.exit(1)

    price_points = []
    
    for tx in transactions:
        price_data = parse_transaction(tx, sol_price_usd, total_supply)
        if price_data:
            price_points.append(price_data)

    # Save results
    with open('price_history_simple.json', 'w') as f:
        json.dump(price_points, f, indent=2)
        print(f"Saved {len(price_points)} price points to price_history_simple.json")

if __name__ == "__main__":
    main()