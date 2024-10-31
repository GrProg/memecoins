# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.
import os
import json
import re
from pathlib import Path
import asyncio
from jupyter import get_all_token_transactions
from tryv3 import process_transaction_file
import shutil

class DataPipelineCoordinator:
    def __init__(self, test_folder='test'):
        self.test_folder = test_folder
        
    def extract_token_address(self, filename: str) -> str:
        """Extract token address from filename"""
        match = re.search(r'price_history_([A-Za-z0-9]{32,44})_', filename)
        if match:
            return match.group(1)
        return None
        
    async def process_single_token(self, filename: str, token_address: str):
        """Process a single token through the pipeline"""
        try:
            print(f"\nProcessing token: {token_address}")
            print("1. Fetching transactions...")
            
            # Get transactions using jupyter.py
            transactions = get_all_token_transactions(token_address)
            if not transactions:
                print("❌ No transactions found")
                return False
            
            # Move the transaction file from jupyter.py output to test folder
            # Find the most recent transaction file
            transaction_files = [f for f in os.listdir() if f.startswith(f'all_transactions_{token_address[:8]}')]
            if not transaction_files:
                print("❌ Transaction file not found")
                return False
                
            # Sort by creation time and get the most recent
            latest_tx_file = max(transaction_files, key=lambda f: os.path.getctime(f))
            
            # Create new filename in test folder
            tx_filename = f"transactions_{token_address}_{Path(filename).stem.split('_')[-2]}_{Path(filename).stem.split('_')[-1]}.json"
            tx_path = os.path.join(self.test_folder, tx_filename)
            
            # Move file to test folder
            shutil.move(latest_tx_file, tx_path)
            print(f"✓ Saved transactions to {tx_filename}")
            
            # Process with tryv3
            print("2. Processing enhanced metrics...")
            enhanced_metrics = process_transaction_file(tx_path)
            
            if not enhanced_metrics:
                print("❌ No enhanced metrics generated")
                return False
                
            # Save enhanced metrics
            enhanced_filename = f"enhanced_{token_address}_{Path(filename).stem.split('_')[-2]}_{Path(filename).stem.split('_')[-1]}.json"
            enhanced_path = os.path.join(self.test_folder, enhanced_filename)
            
            with open(enhanced_path, 'w') as f:
                json.dump(enhanced_metrics, f, indent=2)
                
            print(f"✓ Saved enhanced metrics with {len(enhanced_metrics)} windows")
            return True
            
        except Exception as e:
            print(f"❌ Error processing token {token_address}: {str(e)}")
            return False
    
    async def process_all_files(self):
        """Process all files in the test folder"""
        if not os.path.exists(self.test_folder):
            print(f"Error: Test folder '{self.test_folder}' not found")
            return
            
        # Get all price history files
        price_files = [f for f in os.listdir(self.test_folder) 
                      if f.startswith('price_history_') and f.endswith('.json')]
        
        if not price_files:
            print("No price history files found in test folder")
            return
            
        print(f"\nFound {len(price_files)} price history files to process")
        
        processed = 0
        errors = 0
        
        for filename in price_files:
            token_address = self.extract_token_address(filename)
            
            if not token_address:
                print(f"\n❌ Could not extract token address from {filename}")
                errors += 1
                continue
                
            print(f"\n=== Processing {filename} ===")
            
            # Check if files already exist
            tx_exists = any(f.startswith(f"transactions_{token_address}_") for f in os.listdir(self.test_folder))
            enhanced_exists = any(f.startswith(f"enhanced_{token_address}_") for f in os.listdir(self.test_folder))
            
            if tx_exists and enhanced_exists:
                print("✓ Files already exist for this token, skipping...")
                processed += 1
                continue
            
            success = await self.process_single_token(filename, token_address)
            
            if success:
                processed += 1
            else:
                errors += 1
                
            # Add a small delay between tokens to avoid rate limiting
            await asyncio.sleep(2)
        
        print("\n=== Pipeline Complete ===")
        print(f"Successfully processed: {processed}")
        print(f"Errors: {errors}")
        print(f"Total files: {len(price_files)}")

async def main():
    coordinator = DataPipelineCoordinator()
    await coordinator.process_all_files()

if __name__ == "__main__":
    asyncio.run(main())