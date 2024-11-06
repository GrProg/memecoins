import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, List, Any

class HeliusTransformer:
    def __init__(self, sol_price: float = 177.0, min_sol_threshold: float = 0.1, window_seconds: int = 20):
        """
        Initialize transformer with SOL price in USD and minimum SOL threshold
        
        Args:
            sol_price: Current price of SOL in USD
            min_sol_threshold: Minimum SOL amount to consider a transaction valid
            window_seconds: Number of seconds per time window
        """
        self.sol_price = sol_price
        self.min_sol_threshold = min_sol_threshold
        self.window_seconds = window_seconds
        self.total_supply = 1_000_000_000  # Fixed at 1 billion
        
    def _get_time_windows(self, transactions: List[Dict]) -> List[List[Dict]]:
        """Split transactions into time windows"""
        if not transactions:
            return []
            
        # Sort by timestamp descending (newest first)
        sorted_txs = sorted(transactions, key=lambda x: x['timestamp'], reverse=True)
        
        windows = []
        current_window = []
        window_start = None
        
        for tx in sorted_txs:
            timestamp = tx['timestamp']
            
            if window_start is None:
                window_start = timestamp
                current_window = [tx]
            elif timestamp > window_start - self.window_seconds:
                current_window.append(tx)
            else:
                if current_window:
                    windows.append((window_start, current_window))
                window_start = timestamp
                current_window = [tx]
                
        if current_window:
            windows.append((window_start, current_window))
            
        return windows
        
    def transform_helius_data(self, transactions: List[Dict[str, Any]]) -> tuple:
        """Transform Helius transaction data into price history and enhanced metrics"""
        # Filter transactions by SOL threshold
        valid_txs = [tx for tx in transactions if self._is_valid_transaction(tx)]
        
        # Generate time windows
        enhanced_data = []
        time_windows = self._get_time_windows(valid_txs)
        
        for window_start, window_txs in time_windows:
            window_end = window_start - self.window_seconds
            
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
                'end_timestamp': window_end,
                'window_metrics': window_metrics
            })
        
        # Generate price history for all transactions
        price_history = []
        sorted_txs = sorted(valid_txs, key=lambda x: x['timestamp'], reverse=True)
        
        for tx in sorted_txs:
            if 'tokenTransfers' in tx and tx['tokenTransfers']:
                transfer = tx['tokenTransfers'][0]
                token_amount = float(transfer.get('tokenAmount', 0))
                sol_amount = sum(nt['amount'] for nt in tx['nativeTransfers']) / 1e9
                
                if token_amount > 0:
                    price_in_sol = sol_amount / token_amount
                    price_in_usd = price_in_sol * self.sol_price
                    market_cap = self.total_supply * price_in_usd
                    
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
        
        return price_history, enhanced_data
    
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
            'tx_density': transfer_count / (self.window_seconds if self.window_seconds > 0 else 1)
        }
    
    def _calculate_sol_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate SOL metrics for a time window"""
        sol_volumes = []
        total_transfers = 0
        
        for tx in transactions:
            for transfer in tx.get('nativeTransfers', []):
                amount = transfer['amount'] / 1e9
                sol_volumes.append(amount)
                total_transfers += 1
        
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
        
        interactions = list(account_interactions.values())
        
        return {
            'total_accounts': len(account_interactions),
            'avg_interactions_per_account': np.mean(interactions) if interactions else 0,
            'max_account_interactions': max(interactions) if interactions else 0,
            'interaction_concentration': np.std(interactions) if len(interactions) > 1 else 0
        }
    
    # [Rest of the methods remain the same...]

    def _is_valid_transaction(self, tx: Dict[str, Any]) -> bool:
        """Check if transaction meets minimum SOL threshold"""
        if 'nativeTransfers' not in tx:
            return False
            
        sol_amount = sum(nt['amount'] for nt in tx['nativeTransfers']) / 1e9
        return sol_amount >= self.min_sol_threshold
        
    def _calculate_price_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate price metrics for a time window"""
        prices = []
        mcaps = []
        
        for tx in transactions:
            if 'tokenTransfers' in tx and tx['tokenTransfers']:
                transfer = tx['tokenTransfers'][0]
                token_amount = float(transfer.get('tokenAmount', 0))
                sol_amount = sum(nt['amount'] for nt in tx['nativeTransfers']) / 1e9
                
                if token_amount > 0:
                    price = sol_amount / token_amount
                    prices.append(price)
                    mcap = price * self.sol_price * self.total_supply
                    mcaps.append(mcap)
        
        return {
            'price_start': prices[0] if prices else 0,
            'price_end': prices[-1] if prices else 0,
            'price_change': ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 else 0,
            'price_volatility': np.std(prices) if len(prices) > 1 else 0,
            'mcap_change': ((mcaps[-1] - mcaps[0]) / mcaps[0] * 100) if len(mcaps) > 1 else 0,
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
            if not self._is_valid_transaction(tx):
                continue
                
            if tx.get('type') == 'BURN':
                burns += 1
                if 'tokenTransfers' in tx and tx['tokenTransfers']:
                    total_burned += float(tx['tokenTransfers'][0].get('tokenAmount', 0))
            elif tx.get('type') == 'MINT':
                mints += 1
                if 'tokenTransfers' in tx and tx['tokenTransfers']:
                    total_minted += float(tx['tokenTransfers'][0].get('tokenAmount', 0))
        
        return {
            'burn_count': burns,
            'mint_count': mints,
            'total_burned': total_burned,
            'total_minted': total_minted,
            'net_supply_change': total_minted - total_burned
        }

    def _calculate_fee_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate fee metrics for a time window"""
        fees = [tx.get('fee', 0) / 1e9 for tx in transactions]
        
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
                    program = inst['programId']
                    program_calls[program] = program_calls.get(program, 0) + 1
                    
                # Count inner instructions as well
                for inst in tx['instructions']:
                    if 'innerInstructions' in inst:
                        for inner in inst['innerInstructions']:
                            program = inner['programId']
                            program_calls[program] = program_calls.get(program, 0) + 1
        
        return {
            'unique_programs': len(program_calls),
            'total_program_calls': sum(program_calls.values()),
            'avg_calls_per_program': np.mean(list(program_calls.values())) if program_calls else 0
        }

def convert_helius_data(input_dir: str = 'raw', output_dir: str = 'converted'):
    """Convert Helius transaction data from input directory to price history and enhanced metrics"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize transformer
    transformer = HeliusTransformer(sol_price=177.0, min_sol_threshold=0.1)
    
    # Process all JSON files in input directory
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
            
        print(f"\nProcessing {filename}...")
        
        try:
            # Read input file
            with open(os.path.join(input_dir, filename), 'r') as f:
                transactions = json.load(f)
            
            # Find the full token address from first transaction
            full_address = None
            for tx in transactions:
                if 'tokenTransfers' in tx and tx['tokenTransfers']:
                    full_address = tx['tokenTransfers'][0]['mint']
                    break
                    
            if not full_address:
                print(f"Could not find token address in transactions for {filename}")
                continue
            
            # Transform data
            price_history, enhanced_data = transformer.transform_helius_data(transactions)
            
            if not price_history or not enhanced_data:
                print(f"No valid data extracted from {filename}")
                continue
            
            # Generate output filenames using full address
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            price_filename = f"price_history_{full_address}_{timestamp}.json"
            enhanced_filename = f"enhanced_{full_address}_{timestamp}.json"
            
            # Save converted data
            with open(os.path.join(output_dir, price_filename), 'w') as f:
                json.dump(price_history, f, indent=2)
            
            with open(os.path.join(output_dir, enhanced_filename), 'w') as f:
                json.dump(enhanced_data, f, indent=2)
            
            print(f"Created {price_filename} and {enhanced_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print("\nConversion complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Helius transaction data')
    parser.add_argument('--input', default='all', help='Input directory containing raw transaction files')
    parser.add_argument('--output', default='testt/yes', help='Output directory for converted files')
    
    args = parser.parse_args()
    
    print(f"Converting files from {args.input} to {args.output}")
    convert_helius_data(args.input, args.output)