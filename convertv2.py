import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from typing import Dict, List, Any
import re
import glob

class HeliusTransformerV2:
    def __init__(self, sol_price: float = 220.0, window_seconds: int = 10):
        self.sol_price = sol_price
        self.window_seconds = window_seconds
        self.total_supply = 1_000_000_000

    def _get_time_windows(self, transactions: List[Dict]) -> List[tuple]:
        """Split transactions into time windows"""
        if not transactions:
            return []

        # Sort by timestamp ascending
        sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])
        if not sorted_txs:
            return []

        windows = []
        current_window = []
        window_start = sorted_txs[0]['timestamp']

        for tx in sorted_txs:
            if tx['timestamp'] <= window_start + self.window_seconds:
                current_window.append(tx)
            else:
                if current_window:  # Save completed window
                    windows.append((window_start, current_window))
                # Start new window
                window_start = tx['timestamp']
                current_window = [tx]

        # Add the last window if it contains transactions
        if current_window:
            windows.append((window_start, current_window))

        return windows

    def transform_helius_data(self, transactions: List[Dict[str, Any]], token_address: str) -> tuple:
        """Transform Helius transaction data into price history and enhanced metrics"""
        if not transactions:
            return [], []

        # Sort transactions by timestamp
        sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])

        # Generate price history for all transactions
        price_history = self._generate_price_history(sorted_txs, token_address)

        # Generate enhanced data by time windows
        enhanced_data = []
        time_windows = self._get_time_windows(sorted_txs)

        for window_start, window_txs in time_windows:
            window_metrics = {
                'price_metrics': self._calculate_price_metrics(window_txs),
                'supply_metrics': self._calculate_supply_metrics(window_txs),
                'transaction_metrics': self._calculate_transaction_metrics(window_txs),
                'sol_metrics': self._calculate_sol_metrics(window_txs),
                'account_metrics': self._calculate_account_metrics(window_txs),
                'fee_metrics': self._calculate_fee_metrics(window_txs),
                'program_metrics': self._calculate_program_metrics(window_txs)
            }

            enhanced_data.append({
                'timestamp': window_start,
                'end_timestamp': window_start + self.window_seconds,
                'window_metrics': window_metrics
            })

        return price_history, enhanced_data

    def _generate_price_history(self, transactions: List[Dict], target_token: str) -> List[Dict]:
        """Generate price history with accurate market cap calculation"""
        price_history = []

        for tx in transactions:
            sol_amount, token_amount = self._calculate_precise_small_trade(tx, target_token)

            if token_amount > 0 and sol_amount > 0:
                price_in_sol = sol_amount / token_amount
                price_in_usd = price_in_sol * self.sol_price
                market_cap = price_in_usd * self.total_supply

                price_history.append({
                    'timestamp': tx['timestamp'],
                    'date': datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'token_amount': token_amount,
                    'sol_amount': sol_amount,
                    'price_in_sol': price_in_sol,
                    'price_in_usd': price_in_usd,
                    'market_cap': market_cap,
                    'signature': tx['signature']
                })

        return price_history


    def _get_token_amount(self, tx: Dict, target_token: str = None) -> float:
        """Extract token amount from transaction with target token filtering"""
        if 'tokenTransfers' in tx and tx['tokenTransfers']:
            if target_token:
                # Sum amounts only for target token transfers
                return sum(
                    float(transfer.get('tokenAmount', 0)) 
                    for transfer in tx['tokenTransfers']
                    if transfer.get('mint') == target_token
                )
            else:
                # If no target specified, use first transfer (legacy behavior)
                return float(tx['tokenTransfers'][0].get('tokenAmount', 0))
        return 0
    
    def _calculate_precise_small_trade(self, tx: Dict, target_token: str) -> tuple[float, float]:
        """
        Calculate precise SOL amount and token amount for bot trades
        Returns (sol_amount, token_amount)
        """
        actual_sol_amount = 0
        actual_token_amount = 0
    
        # Get token amount from token transfers
        if 'tokenTransfers' in tx:
            target_transfers = [t for t in tx['tokenTransfers'] 
                              if t.get('mint') == target_token]
            if target_transfers:
                actual_token_amount = float(target_transfers[0].get('tokenAmount', 0))
    
        # For bot trading patterns, we need to look at specific transfer patterns
        if 'nativeTransfers' in tx:
            # First, look for transfer to original seller (typically around 0.01-0.02 SOL)
            trading_transfers = []
            for transfer in tx['nativeTransfers']:
                amount = float(transfer['amount']) / 1e9  # Convert lamports to SOL
                # Filter transfers in the typical bot trading range (0.005-0.05 SOL)
                if 0.005 <= amount <= 0.05:
                    trading_transfers.append(amount)
    
            if trading_transfers:
                # For bot trades, use the most likely trading amount
                actual_sol_amount = max(trading_transfers)  # Usually the largest in the valid range
            else:
                # Fallback: Look for any reasonable trading amount
                all_transfers = [
                    float(t['amount']) / 1e9 
                    for t in tx['nativeTransfers'] 
                    if float(t['amount']) / 1e9 >= 0.001  # Minimum threshold
                ]
                if all_transfers:
                    # Use the most likely trading amount
                    actual_sol_amount = min(t for t in all_transfers if t >= 0.001)
    
        return actual_sol_amount, actual_token_amount

    def _get_sol_amount(self, tx: Dict) -> float:
        """Extract SOL amount for standard trades"""
        try:
            if 'nativeTransfers' in tx:
                fee = float(tx.get('fee', 0)) / 1e9

                # First try to identify the bonding curve transfer
                # These usually go to addresses containing 'Curve' or specific program addresses
                bonding_transfers = []
                for nt in tx['nativeTransfers']:
                    amount = float(nt['amount']) / 1e9
                    if amount > 0.005 and abs(amount - fee) > 0.000001:
                        # Look for transfers to bonding curve or program accounts
                        if 'toUserAccount' in nt:
                            bonding_transfers.append(amount)

                # If we found bonding curve transfers, use the most likely swap amount
                if bonding_transfers:
                    # For pump trades, the actual swap is usually between 0.01-0.02 SOL
                    # Filter to this range if such transfers exist
                    swap_range_transfers = [t for t in bonding_transfers if 0.008 <= t <= 0.025]
                    if swap_range_transfers:
                        return min(swap_range_transfers)
                    # If no transfers in the typical range, take the smallest valid transfer
                    return min(bonding_transfers)

            # Fallback to wrapped SOL check
            if 'tokenTransfers' in tx:
                wrapped_transfers = [
                    float(t.get('tokenAmount', 0))
                    for t in tx['tokenTransfers']
                    if t.get('mint') == 'So11111111111111111111111111111111111111112'
                    and float(t.get('tokenAmount', 0)) >= 0.008  # Minimum threshold
                ]
                if wrapped_transfers:
                    # Same logic - prefer amounts in the typical range
                    swap_range_transfers = [t for t in wrapped_transfers if 0.008 <= t <= 0.025]
                    if swap_range_transfers:
                        return min(swap_range_transfers)
                    return min(wrapped_transfers)

        except Exception as e:
            print(f"Error in _get_sol_amount: {str(e)}")

        return 0

    def _calculate_price_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate price metrics for a time window with zero handling"""
        prices = []
        mcaps = []

        for tx in transactions:
            if 'tokenTransfers' in tx and tx['tokenTransfers']:
                token_amount = self._get_token_amount(tx)
                sol_amount = self._get_sol_amount(tx)

                if token_amount > 0 and sol_amount > 0:  # Only calculate price if amounts are valid
                    price = sol_amount / token_amount
                    prices.append(price)
                    mcap = price * self.sol_price * self.total_supply
                    mcaps.append(mcap)

        # Handle empty price lists
        if not prices:
            return {
                'price_start': 0,
                'price_end': 0,
                'price_change': 0,
                'price_volatility': 0,
                'mcap_change': 0,
                'mcap_volatility': 0,
                'valid_price_points': 0,
                'valid_mcap_points': 0
            }

        # Calculate metrics only if we have valid prices
        price_change = ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[0] != 0 else 0
        mcap_change = ((mcaps[-1] - mcaps[0]) / mcaps[0] * 100) if len(mcaps) > 1 and mcaps[0] != 0 else 0

        return {
            'price_start': prices[0],
            'price_end': prices[-1],
            'price_change': price_change,
            'price_volatility': np.std(prices) if len(prices) > 1 else 0,
            'mcap_change': mcap_change,
            'mcap_volatility': np.std(mcaps) if len(mcaps) > 1 else 0,
            'valid_price_points': len(prices),
            'valid_mcap_points': len(mcaps)
        }


    def _calculate_supply_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate supply metrics for a time window"""
        burns = 0
        mints = 0
        total_burned = 0
        total_minted = 0

        for tx in transactions:
            if tx.get('type') == 'BURN':
                burns += 1
                total_burned += self._get_token_amount(tx)
            elif tx.get('type') == 'MINT':
                mints += 1
                total_minted += self._get_token_amount(tx)

        return {
            'burn_count': burns,
            'mint_count': mints,
            'total_burned': total_burned,
            'total_minted': total_minted,
            'net_supply_change': total_minted - total_burned
        }

    def _calculate_transaction_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate transaction metrics for a time window"""
        total_txs = len(transactions)
        transfer_count = sum(1 for tx in transactions if 'tokenTransfers' in tx and tx['tokenTransfers'])

        unique_accounts = set()
        for tx in transactions:
            if 'accountData' in tx:
                for account in tx['accountData']:
                    unique_accounts.add(account['account'])

        return {
            'transfer_count': transfer_count,
            'total_transactions': total_txs,
            'unique_accounts': len(unique_accounts),
            'tx_density': total_txs / self.window_seconds if self.window_seconds > 0 else 0
        }

    def _calculate_sol_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate SOL metrics for a time window"""
        sol_volumes = [self._get_sol_amount(tx) for tx in transactions]
        total_transfers = len(sol_volumes)

        return {
            'total_sol_volume': sum(sol_volumes),
            'avg_sol_per_tx': np.mean(sol_volumes) if sol_volumes else 0,
            'sol_tx_count': total_transfers,
            'sol_flow_volatility': np.std(sol_volumes) if len(sol_volumes) > 1 else 0
        }

    def _calculate_account_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate account metrics for a time window"""
        account_interactions = {}

        for tx in transactions:
            if 'accountData' in tx:
                for account in tx['accountData']:
                    acc = account['account']
                    account_interactions[acc] = account_interactions.get(acc, 0) + 1

        interactions = list(account_interactions.values()) if account_interactions else [0]

        return {
            'total_accounts': len(account_interactions),
            'avg_interactions_per_account': np.mean(interactions),
            'max_account_interactions': max(interactions),
            'interaction_concentration': np.std(interactions) if len(interactions) > 1 else 0
        }

    def _calculate_fee_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate fee metrics for a time window"""
        fees = [float(tx.get('fee', 0)) / 1e9 for tx in transactions]

        return {
            'total_fees': sum(fees),
            'avg_fee': np.mean(fees) if fees else 0,
            'max_fee': max(fees) if fees else 0,
            'fee_volatility': np.std(fees) if len(fees) > 1 else 0
        }

    def _calculate_program_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate program metrics for a time window"""
        program_calls = {}

        for tx in transactions:
            if 'instructions' in tx:
                for inst in tx['instructions']:
                    program = inst.get('programId')
                    if program:
                        program_calls[program] = program_calls.get(program, 0) + 1

                    # Count inner instructions
                    if 'innerInstructions' in inst:
                        for inner in inst['innerInstructions']:
                            program = inner.get('programId')
                            if program:
                                program_calls[program] = program_calls.get(program, 0) + 1

        calls = list(program_calls.values()) if program_calls else [0]

        return {
            'unique_programs': len(program_calls),
            'total_program_calls': sum(calls),
            'avg_calls_per_program': np.mean(calls)
        }

def log_filtered_transactions(filtered_tx, original_address, reason, log_dir):
    """Log filtered transactions to a file"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"filtered_transactions_{original_address}.json")
    
    log_entry = {
        "timestamp": filtered_tx['timestamp'],
        "signature": filtered_tx['signature'],
        "reason": reason,
        "details": filtered_tx
    }
    
    with open(log_file, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')  # Add a newline for readability

def clean_transaction_data(transactions, 
                           original_address,
                           log_dir,
                           min_market_cap=4999,
                           max_market_cap_allowed=100000,  # Increased to allow for higher caps
                           min_sol_amount=0.0039,  # Lowered to catch smaller trades
                           initial_seconds=10):
    
    # Sort transactions by timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    
    # Get the timestamp of the first transaction
    initial_timestamp = transactions[0]['timestamp']
    
    cleaned_transactions = []
    max_market_cap = 0
    
    for tx in transactions:
        # Log and skip transactions with too little SOL or zero token amount
        if tx['sol_amount'] < min_sol_amount:
            log_filtered_transactions(tx, original_address, "Insufficient SOL amount", log_dir)
            continue
        if tx['token_amount'] <= 0:
            log_filtered_transactions(tx, original_address, "Zero token amount", log_dir)
            continue
            
        # If transaction is within initial_seconds of first transaction,
        # set market cap to 5000
        if tx['timestamp'] - initial_timestamp <= initial_seconds:
            tx['market_cap'] = 5000
            tx['price_in_usd'] = 5000 / tx['token_amount'] if tx['token_amount'] > 0 else 0
        
        # Only process transactions within the allowed market cap range
        if min_market_cap <= tx['market_cap'] <= max_market_cap_allowed:
            cleaned_transactions.append(tx)
            # Update max_market_cap
            max_market_cap = max(max_market_cap, tx['market_cap'])
        else:
            log_filtered_transactions(tx, original_address, "Market cap out of range", log_dir)
    
    # If we have no valid transactions after cleaning, return dummy data
    if not cleaned_transactions:
        return [], 0
    
    return cleaned_transactions, max_market_cap

def convert_helius_data_v2(input_dir: str = 'all', output_dir: str = 'yes', log_dir: str = 'test'):
    """Convert Helius transaction data from input directory to price history and enhanced metrics"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    transformer = HeliusTransformerV2()

    for filename in os.listdir(input_dir):
        if not filename.startswith('transactions_') or not filename.endswith('.json'):
            continue

        print(f"\nProcessing {filename}...")
        file_path = os.path.join(input_dir, filename)

        try:
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                print(f"Skipping empty file: {filename}")
                continue

            # Read input file with error handling
            try:
                with open(file_path, 'r') as f:
                    file_content = f.read().strip()
                    if not file_content:
                        print(f"Empty content in file: {filename}")
                        continue
                    transactions = json.loads(file_content)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in file {filename}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
                continue

            # Validate transactions data
            if not isinstance(transactions, list):
                print(f"Invalid data format in {filename} - expected list of transactions")
                continue

            if not transactions:
                print(f"No transactions found in {filename}")
                continue

            # Extract token address from filename
            try:
                original_address = filename.split('_')[1].split('.')[0]
            except IndexError:
                print(f"Invalid filename format: {filename}")
                continue

            # Transform data using the original address
            try:
                price_history, enhanced_data = transformer.transform_helius_data(transactions, original_address)
                
                if not price_history and not enhanced_data:
                    print(f"No valid data extracted from {filename}")
                    continue

                # Clean and sort price history data
                cleaned_price_history, max_market_cap = clean_transaction_data(price_history, original_address, log_dir)
                if not cleaned_price_history:
                    print(f"No valid transactions after cleaning in {filename}")
                    continue

                cleaned_price_history.sort(key=lambda x: x['timestamp'])

                # Generate output filenames using the original address and max market cap
                price_filename = f"price_history_{original_address}_{int(max_market_cap)}.json"
                enhanced_filename = f"enhanced_{original_address}_{int(max_market_cap)}.json"

                # Save converted data
                output_price_path = os.path.join(output_dir, price_filename)
                output_enhanced_path = os.path.join(output_dir, enhanced_filename)

                with open(output_price_path, 'w') as f:
                    json.dump(cleaned_price_history, f, indent=2)

                with open(output_enhanced_path, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)

                print(f"Created {price_filename} and {enhanced_filename}")

            except Exception as e:
                print(f"Error transforming data for {filename}: {str(e)}")
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"Unexpected error processing {filename}: {str(e)}")
            traceback.print_exc()
            continue

    print("\nConversion complete!")

if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description='Convert Helius transaction data (V2)')
    parser.add_argument('--input', default='all', help='Input directory containing raw transaction files')
    parser.add_argument('--output', default='testt/yes', help='Output directory for converted files')
    parser.add_argument('--log', default='test', help='Directory for logging filtered transactions')

    args = parser.parse_args()

    print(f"Converting files from {args.input} to {args.output}")
    print(f"Logging filtered transactions to {args.log}")
    try:
        convert_helius_data_v2(args.input, args.output, args.log)
    except Exception as e:
        print(f"Fatal error during conversion: {str(e)}")
        traceback.print_exc()
        sys.exit(1)