# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.
import asyncio
import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pump_pipeline.log'),
        logging.StreamHandler()
    ]
)

class PumpPipelineCoordinator:
    def __init__(self):
        self.processed_tokens = set()
        self.ensure_directories()
        self.python_cmd = sys.executable
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs('all', exist_ok=True)
        
    def get_current_pump_file(self):
        """Get the current day's pump file name"""
        return f"pumps_{datetime.now().strftime('%Y%m%d')}.json"

    async def wait_for_file(self, pattern: str, timeout: int = 300) -> Path:
        """Wait for a file matching the pattern to appear and be fully written"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            files = list(Path('.').glob(pattern))
            if files:
                file = files[0]
                # Wait for file size to stabilize (indicating write is complete)
                last_size = -1
                current_size = file.stat().st_size
                while last_size != current_size and time.time() - start_time < timeout:
                    await asyncio.sleep(1)
                    last_size = current_size
                    current_size = file.stat().st_size
                
                if last_size == current_size:
                    return file
            await asyncio.sleep(1)
        raise TimeoutError(f"Timeout waiting for file matching {pattern}")

    async def run_jupyter_script(self, token_address: str) -> bool:
        """Run jupyter.py to fetch transaction data"""
        try:
            logging.info(f"Starting transaction fetch for {token_address}")
            
            cmd = f'"{self.python_cmd}" jupyter.py "{token_address}"'
            logging.info(f"Running command: {cmd}")
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""
            
            if process.returncode != 0:
                logging.error(f"Error running jupyter.py: {stderr_text}")
                return False
                
            logging.info(f"Jupyter output: {stdout_text}")
            
            # Wait for the transaction file to be fully written
            try:
                tx_file = await self.wait_for_file(f'all_transactions_{token_address[:8]}*.json')
                logging.info(f"Found and verified transaction file: {tx_file}")
                return True
            except TimeoutError:
                logging.error("Timeout waiting for transaction file to be written")
                return False
                
        except Exception as e:
            logging.error(f"Exception running jupyter.py: {str(e)}")
            return False
            
    async def run_analysis_script(self, token_address: str) -> bool:
        """Run tryv2.py to analyze transaction data"""
        try:
            logging.info(f"Starting analysis for {token_address}")
            
            # Find and verify the transaction file
            try:
                tx_file = await self.wait_for_file(f'all_transactions_{token_address[:8]}*.json', timeout=30)
            except TimeoutError:
                logging.error(f"No transaction file found for {token_address}")
                return False
                
            # Run tryv2.py with the transaction file as argument
            process = await asyncio.create_subprocess_shell(
                f'"{self.python_cmd}" tryv2.py "{tx_file}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logging.error(f"Error running tryv2.py: {stderr.decode() if stderr else 'Unknown error'}")
                return False
            
            # Wait for the analysis result file
            try:
                await self.wait_for_file('price_history_simple.json', timeout=30)
            except TimeoutError:
                logging.error("Timeout waiting for analysis results file")
                return False
                
            # Move result file to all/ directory
            result_file = 'price_history_simple.json'
            if os.path.exists(result_file):
                new_filename = f"all/price_history_{token_address}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.rename(result_file, new_filename)
                logging.info(f"Moved analysis results to {new_filename}")
                
            # Cleanup transaction json
            tx_file.unlink()
            logging.info(f"Cleaned up {tx_file}")
                
            return True
            
        except Exception as e:
            logging.error(f"Exception running tryv2.py: {str(e)}")
            return False

    async def process_pump(self, pump_data: dict) -> bool:
        """Process a single pump completely"""
        token_address = pump_data['token']
        
        if token_address in self.processed_tokens:
            logging.info(f"Token {token_address} already processed, skipping")
            return True
            
        logging.info(f"Starting complete processing for {token_address}")
        
        # Wait 15 minutes from pump detection
        pump_time = datetime.fromisoformat(pump_data['timestamp'])
        wait_until = pump_time + timedelta(minutes=15)
        now = datetime.now()
        
        if wait_until > now:
            delay = (wait_until - now).total_seconds()
            logging.info(f"Waiting {delay:.0f} seconds before processing {token_address}")
            await asyncio.sleep(delay)
        
        # Run the complete analysis pipeline
        if await self.run_jupyter_script(token_address):
            if await self.run_analysis_script(token_address):
                logging.info(f"Completed processing for {token_address}")
                self.processed_tokens.add(token_address)
                return True
            else:
                logging.error(f"Failed analysis for {token_address}")
                return False
        else:
            logging.error(f"Failed transaction fetch for {token_address}")
            return False

    async def process_pumps_sequentially(self):
        """Process all pumps from the file one by one"""
        while True:
            try:
                pump_file = self.get_current_pump_file()
                if not os.path.exists(pump_file):
                    logging.info(f"Waiting for pump file {pump_file}")
                    await asyncio.sleep(5)
                    continue
                
                # Read all pumps from file
                with open(pump_file, 'r') as f:
                    lines = f.readlines()
                
                # Process each pump sequentially
                for line in lines:
                    try:
                        pump_data = json.loads(line.strip())
                        await self.process_pump(pump_data)
                    except json.JSONDecodeError:
                        logging.error(f"Error parsing JSON line: {line.strip()}")
                        continue
                
                # Wait before checking for new pumps
                await asyncio.sleep(5)
                
            except Exception as e:
                logging.error(f"Error in pump processing: {str(e)}")
                await asyncio.sleep(5)

async def main():
    coordinator = PumpPipelineCoordinator()
    await coordinator.process_pumps_sequentially()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Pipeline coordinator stopped by user")
    except Exception as e:
        logging.error(f"Pipeline coordinator error: {str(e)}")