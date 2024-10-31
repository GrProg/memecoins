# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import os
import shutil
from collections import defaultdict

class TransactionWindowAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
    def analyze_window_transactions(self, transactions: List[Dict]) -> Dict:
        """Analyze transactions within a time window"""
        if not transactions:
            return {}
            
        # Sort by timestamp
        transactions = sorted(transactions, key=lambda x: x['timestamp'])
        
        # Basic price and market cap metrics
        price_metrics = self._calculate_price_metrics(transactions)
        
        # Supply analysis
        supply_metrics = self._analyze_supply_changes(transactions)
        
        # Transaction patterns
        tx_metrics = self._analyze_transaction_patterns(transactions)
        
        # SOL flow analysis
        sol_metrics = self._analyze_sol_flows(transactions)
        
        # Account interaction analysis
        account_metrics = self._analyze_account_patterns(transactions)
        
        # Fee analysis
        fee_metrics = self._analyze_fees(transactions)
        
        # Program interaction analysis
        program_metrics = self._analyze_program_interactions(transactions)
        
        return {
            "timestamp": transactions[0]['timestamp'],
            "end_timestamp": transactions[-1]['timestamp'],
            "window_metrics": {
                "price_metrics": price_metrics,
                "supply_metrics": supply_metrics,
                "transaction_metrics": tx_metrics,
                "sol_metrics": sol_metrics,
                "account_metrics": account_metrics,
                "fee_metrics": fee_metrics,
                "program_metrics": program_metrics
            }
        }

    def _calculate_price_metrics(self, transactions: List[Dict]) -> Dict:
        try:
            # Extract prices and market caps, defaulting to 0 if missing
            prices = []
            market_caps = []

            for tx in transactions:
                # Get price, safely handle missing or invalid values
                try:
                    price = float(tx.get('price_in_usd', 0))
                    prices.append(price if price > 0 else 0)
                except (ValueError, TypeError):
                    prices.append(0)

                # Get market cap, safely handle missing or invalid values
                try:
                    mcap = float(tx.get('market_cap', 0))
                    market_caps.append(mcap if mcap > 0 else 0)
                except (ValueError, TypeError):
                    market_caps.append(0)

            # Calculate price changes if we have valid prices
            if len(prices) > 1 and any(p > 0 for p in prices):
                # Filter out zero prices for change calculation
                valid_prices = [p for p in prices if p > 0]
                if len(valid_prices) > 1:
                    price_changes = np.diff(valid_prices) / valid_prices[:-1]
                else:
                    price_changes = [0]
            else:
                price_changes = [0]

            # Calculate market cap changes if we have valid market caps
            if len(market_caps) > 1 and any(m > 0 for m in market_caps):
                # Filter out zero market caps for change calculation
                valid_mcaps = [m for m in market_caps if m > 0]
                if len(valid_mcaps) > 1:
                    mcap_changes = np.diff(valid_mcaps) / valid_mcaps[:-1]
                else:
                    mcap_changes = [0]
            else:
                mcap_changes = [0]

            return {
                "price_start": prices[0] if prices else 0,
                "price_end": prices[-1] if prices else 0,
                "price_change": float(np.mean(price_changes)) if price_changes else 0,
                "price_volatility": float(np.std(price_changes)) if len(price_changes) > 1 else 0,
                "mcap_change": float(np.mean(mcap_changes)) if mcap_changes else 0,
                "mcap_volatility": float(np.std(mcap_changes)) if len(mcap_changes) > 1 else 0,
                "valid_price_points": sum(1 for p in prices if p > 0),
                "valid_mcap_points": sum(1 for m in market_caps if m > 0)
            }
        except Exception as e:
            # Fallback to zeros if anything goes wrong
            print(f"Error calculating price metrics: {str(e)}")
            return {
                "price_start": 0,
                "price_end": 0,
                "price_change": 0,
                "price_volatility": 0,
                "mcap_change": 0,
                "mcap_volatility": 0,
                "valid_price_points": 0,
                "valid_mcap_points": 0
            }
    
    def _analyze_supply_changes(self, transactions: List[Dict]) -> Dict:
        """Analyze token supply changes"""
        burns = [tx for tx in transactions if tx.get('type') == 'BURN']
        mints = [tx for tx in transactions if tx.get('type') == 'MINT']
        
        total_burned = sum(float(tx['tokenTransfers'][0]['tokenAmount']) for tx in burns if 'tokenTransfers' in tx)
        total_minted = sum(float(tx['tokenTransfers'][0]['tokenAmount']) for tx in mints if 'tokenTransfers' in tx)
        
        return {
            "burn_count": len(burns),
            "mint_count": len(mints),
            "total_burned": total_burned,
            "total_minted": total_minted,
            "net_supply_change": total_minted - total_burned
        }
    
    def _analyze_transaction_patterns(self, transactions: List[Dict]) -> Dict:
        """Analyze transaction patterns"""
        transfers = [tx for tx in transactions if tx.get('type') == 'TRANSFER']
        
        # Calculate unique accounts involved in transactions
        unique_accounts = set()
        for tx in transactions:
            for transfer in tx.get('tokenTransfers', []):
                if 'fromUserAccount' in transfer:
                    unique_accounts.add(transfer['fromUserAccount'])
                if 'toUserAccount' in transfer:
                    unique_accounts.add(transfer['toUserAccount'])
        
        return {
            "transfer_count": len(transfers),
            "total_transactions": len(transactions),
            "unique_accounts": len(unique_accounts),
            "tx_density": len(transactions) / self.window_size if self.window_size > 0 else 0
        }
    
    def _analyze_sol_flows(self, transactions: List[Dict]) -> Dict:
        """Analyze SOL token flows"""
        sol_transfers = []
        for tx in transactions:
            for transfer in tx.get('nativeTransfers', []):
                sol_transfers.append(float(transfer['amount']) / 1e9)  # Convert lamports to SOL
        
        return {
            "total_sol_volume": sum(sol_transfers),
            "avg_sol_per_tx": np.mean(sol_transfers) if sol_transfers else 0,
            "sol_tx_count": len(sol_transfers),
            "sol_flow_volatility": float(np.std(sol_transfers)) if len(sol_transfers) > 1 else 0
        }
    
    def _analyze_account_patterns(self, transactions: List[Dict]) -> Dict:
        """Analyze account interaction patterns"""
        account_interactions = defaultdict(int)
        for tx in transactions:
            for transfer in tx.get('tokenTransfers', []):
                if 'fromUserAccount' in transfer:
                    account_interactions[transfer['fromUserAccount']] += 1
                if 'toUserAccount' in transfer:
                    account_interactions[transfer['toUserAccount']] += 1
        
        interaction_counts = list(account_interactions.values())
        
        return {
            "total_accounts": len(account_interactions),
            "avg_interactions_per_account": np.mean(interaction_counts) if interaction_counts else 0,
            "max_account_interactions": max(interaction_counts) if interaction_counts else 0,
            "interaction_concentration": float(np.std(interaction_counts)) if len(interaction_counts) > 1 else 0
        }
    
    def _analyze_fees(self, transactions: List[Dict]) -> Dict:
        """Analyze transaction fees"""
        fees = [float(tx.get('fee', 0)) / 1e9 for tx in transactions]  # Convert lamports to SOL
        
        return {
            "total_fees": sum(fees),
            "avg_fee": np.mean(fees) if fees else 0,
            "max_fee": max(fees) if fees else 0,
            "fee_volatility": float(np.std(fees)) if len(fees) > 1 else 0
        }
    
    def _analyze_program_interactions(self, transactions: List[Dict]) -> Dict:
        """Analyze program interactions"""
        program_calls = defaultdict(int)
        for tx in transactions:
            for instruction in tx.get('instructions', []):
                if 'programId' in instruction:
                    program_calls[instruction['programId']] += 1
        
        return {
            "unique_programs": len(program_calls),
            "total_program_calls": sum(program_calls.values()),
            "avg_calls_per_program": np.mean(list(program_calls.values())) if program_calls else 0
        }

class EnhancedPumpDetector:
    def __init__(self,
                 window_size: int = 20,
                 volatility_threshold: float = 0.4,
                 min_trend_duration: int = 60,
                 dump_threshold: float = 15):
        
        self.window_size = window_size
        self.volatility_threshold = volatility_threshold
        self.min_trend_duration = min_trend_duration
        self.dump_threshold = dump_threshold
        self.analyzer = TransactionWindowAnalyzer(window_size)
        
    def analyze_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict]:
        """
        Analyze transactions and create enhanced metrics for each time window
        """
        if not transactions:
            return []
            
        # Sort transactions by timestamp
        transactions = sorted(transactions, key=lambda x: x['timestamp'])
        
        # Group transactions into windows
        windows = []
        current_window = []
        window_start = transactions[0]['timestamp']
        
        for tx in transactions:
            if tx['timestamp'] - window_start <= self.window_size:
                current_window.append(tx)
            else:
                if current_window:
                    windows.append(current_window)
                current_window = [tx]
                window_start = tx['timestamp']
        
        if current_window:
            windows.append(current_window)
        
        # Analyze each window
        window_metrics = []
        for window_txs in windows:
            metrics = self.analyzer.analyze_window_transactions(window_txs)
            window_metrics.append(metrics)
        
        return window_metrics

def process_transaction_file(file_path: str) -> List[Dict]:
    """Process a single transaction file and return enhanced metrics"""
    try:
        with open(file_path, 'r') as f:
            transactions = json.load(f)
            
        detector = EnhancedPumpDetector()
        return detector.analyze_transactions(transactions)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def process_all_files(input_dir: str = 'inputForTry', output_dir: str = 'enhanced'):
    """Process all transaction files and save enhanced metrics"""
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    files_processed = 0
    errors = 0
    
    print(f"\nProcessing files from '{input_dir}' directory...")
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"enhanced_{filename}")
            
            try:
                print(f"\nProcessing {filename}...")
                enhanced_metrics = process_transaction_file(input_path)
                
                if enhanced_metrics:
                    with open(output_path, 'w') as f:
                        json.dump(enhanced_metrics, f, indent=2)
                    
                    files_processed += 1
                    print(f"✓ Successfully processed {filename}: {len(enhanced_metrics)} windows")
                else:
                    errors += 1
                    print(f"✗ No metrics generated for {filename}")
                
            except Exception as e:
                errors += 1
                print(f"✗ Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing complete:")
    print(f"Total files processed successfully: {files_processed}")
    print(f"Total errors: {errors}")
    print(f"Enhanced metrics saved to '{output_dir}' directory")

if __name__ == "__main__":
    process_all_files()