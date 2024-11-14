import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, List, Any

class BotTradeTransformer:
    def __init__(self, sol_price: float = 213.0):
        self.sol_price = sol_price
        self.total_supply = 1_000_000_000
        
    def _get_time_windows(self, transactions: List[Dict]) -> List[Dict]:
        """Group transactions into 10-second windows"""
        if not transactions:
            return []
            
        # Sort by timestamp
        sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])
        windows = []
        current_window = []
        window_start = sorted_txs[0]['timestamp']
        
        for tx in sorted_txs:
            if tx['timestamp'] - window_start <= 10:  # 10-second windows
                current_window.append(tx)
            else:
                if current_window:
                    windows.append({
                        'start_time': window_start,
                        'transactions': current_window
                    })
                window_start = tx['timestamp']
                current_window = [tx]
                
        if current_window:
            windows.append({
                'start_time': window_start,
                'transactions': current_window
            })
            
        return windows

    def _extract_bot_trade_amounts(self, tx: Dict) -> tuple[float, float, str, str]:
        """
        Extract token and SOL amounts from bot trades by looking for specific patterns:
        - Transfer to seller with matching 1% fee
        - Token transfers with specific patterns
        Returns (token_amount, sol_amount, seller_address, buyer_address)
        """
        token_amount = 0
        sol_amount = 0
        seller_address = ""
        buyer_address = ""
        
        # Get token transfer info
        if 'tokenTransfers' in tx and tx['tokenTransfers']:
            token_transfer = tx['tokenTransfers'][0]
            token_amount = float(token_transfer.get('tokenAmount', 0))
            seller_address = token_transfer.get('fromUserAccount', '')
            buyer_address = token_transfer.get('toUserAccount', '')
        
        # Find main trade amount by looking for transfer to seller with 1% fee
        if 'nativeTransfers' in tx:
            transfers = tx['nativeTransfers']
            for i, transfer in enumerate(transfers):
                amount = float(transfer['amount']) / 1e9  # Convert lamports to SOL
                # Look for seller transfer
                if transfer.get('toUserAccount') == seller_address:
                    # Check next transfer for 1% fee
                    if i + 1 < len(transfers):
                        next_amount = float(transfers[i + 1]['amount']) / 1e9
                        # Verify 1% fee relationship
                        if abs(next_amount - (amount * 0.01)) < 0.0001:
                            sol_amount = amount
                            break
        
        return token_amount, sol_amount, seller_address, buyer_address

    def transform_transactions(self, transactions: List[Dict], target_token: str) -> tuple[List[Dict], List[Dict]]:
        """Transform bot trading transactions into price history and metrics"""
        price_history = []
        enhanced_data = []
        
        # Process transactions
        windows = self._get_time_windows(transactions)
        
        for window in windows:
            window_metrics = {
                'trades': [],
                'total_volume': 0,
                'unique_buyers': set(),
                'unique_sellers': set(),
            }
            
            for tx in window['transactions']:
                token_amount, sol_amount, seller, buyer = self._extract_bot_trade_amounts(tx)
                
                if token_amount > 0 and sol_amount > 0:
                    price_in_sol = sol_amount / token_amount
                    price_in_usd = price_in_sol * self.sol_price
                    market_cap = price_in_usd * self.total_supply
                    
                    trade_data = {
                        'timestamp': tx['timestamp'],
                        'date': datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                        'token_amount': token_amount,
                        'sol_amount': sol_amount,
                        'price_in_sol': price_in_sol,
                        'price_in_usd': price_in_usd,
                        'market_cap': market_cap,
                        'signature': tx['signature'],
                        'seller': seller,
                        'buyer': buyer,
                    }
                    
                    price_history.append(trade_data)
                    window_metrics['trades'].append(trade_data)
                    window_metrics['total_volume'] += sol_amount
                    window_metrics['unique_buyers'].add(buyer)
                    window_metrics['unique_sellers'].add(seller)
            
            if window_metrics['trades']:
                # Calculate window statistics
                prices = [t['price_in_sol'] for t in window_metrics['trades']]
                enhanced_data.append({
                    'timestamp': window['start_time'],
                    'end_timestamp': window['start_time'] + 10,
                    'window_metrics': {
                        'price_metrics': {
                            'price_start': prices[0],
                            'price_end': prices[-1],
                            'price_change': ((prices[-1] - prices[0]) / prices[0] * 100) if prices[0] != 0 else 0,
                            'price_volatility': np.std(prices) if len(prices) > 1 else 0,
                            'valid_price_points': len(prices)
                        },
                        'volume_metrics': {
                            'total_sol_volume': window_metrics['total_volume'],
                            'trade_count': len(window_metrics['trades']),
                            'avg_trade_size': window_metrics['total_volume'] / len(window_metrics['trades'])
                        },
                        'participant_metrics': {
                            'unique_buyers': len(window_metrics['unique_buyers']),
                            'unique_sellers': len(window_metrics['unique_sellers']),
                            'total_participants': len(window_metrics['unique_buyers'] | window_metrics['unique_sellers'])
                        }
                    }
                })
                
        return price_history, enhanced_data

def convert_bot_trades(input_dir: str = 'all', output_dir: str = 'converted') -> None:
    """Process bot trading data from input directory"""
    os.makedirs(output_dir, exist_ok=True)
    transformer = BotTradeTransformer()
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
            
        print(f"\nProcessing {filename}")
        input_path = os.path.join(input_dir, filename)
        
        try:
            with open(input_path, 'r') as f:
                transactions = json.load(f)
                
            # Extract token address from first valid transaction
            token_address = None
            for tx in transactions:
                if 'tokenTransfers' in tx and tx['tokenTransfers']:
                    token_address = tx['tokenTransfers'][0].get('mint')
                    break
                    
            if not token_address:
                print(f"No token address found in {filename}")
                continue
                
            # Transform data
            price_history, enhanced_data = transformer.transform_transactions(transactions, token_address)
            
            if not price_history or not enhanced_data:
                print(f"No valid trades found in {filename}")
                continue
                
            # Save files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            max_market_cap = max(trade['market_cap'] for trade in price_history)
            
            price_file = f"price_history_{token_address}_{int(max_market_cap)}.json"
            enhanced_file = f"enhanced_{token_address}_{int(max_market_cap)}.json"
            
            with open(os.path.join(output_dir, price_file), 'w') as f:
                json.dump(price_history, f, indent=2)
                
            with open(os.path.join(output_dir, enhanced_file), 'w') as f:
                json.dump(enhanced_data, f, indent=2)
                
            print(f"Created {price_file} and {enhanced_file}")
            print(f"Found {len(price_history)} valid bot trades")
            print(f"Max market cap: ${int(max_market_cap):,}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert bot trading data')
    parser.add_argument('--input', default='all', help='Input directory')
    parser.add_argument('--output', default='testt/yes', help='Output directory')
    
    args = parser.parse_args()
    convert_bot_trades(args.input, args.output)