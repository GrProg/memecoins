import os
import glob
import re
import subprocess
import shutil
import time
from typing import Optional, List, Tuple
import logging
from datetime import datetime
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'clean_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DataCleaner:
    def __init__(self):
        self.base_dir = os.getcwd()
        self.newyes_dir = os.path.join('pameligo', 'newyes')
        self.newyesv2_dir = os.path.join('pameligo', 'newyesV2')
        self.all_dir = 'all'
        self.testt_yes_dir = os.path.join('testt', 'yes')
        
        # Ensure all directories exist
        for directory in [self.newyesv2_dir, self.all_dir, self.testt_yes_dir]:
            os.makedirs(directory, exist_ok=True)
        
    def extract_address_from_filename(self, filename: str) -> Optional[str]:
        """Extract token address from enhanced filename."""
        match = re.search(r'enhanced_([a-zA-Z0-9]{32,44})_\d+\.json', filename)
        return match.group(1) if match else None
        
    def get_enhanced_files(self) -> List[str]:
        """Get all enhanced files from newyes directory."""
        pattern = os.path.join(self.newyes_dir, 'enhanced_*.json')
        return glob.glob(pattern)
        
    def run_jupyter(self, address: str) -> bool:
        """Run jupyter script for an address."""
        try:
            logging.info(f"Running jupyter for address: {address}")
            result = subprocess.run(
                ['python', 'jupyterV2.py', address],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info("Jupyter completed successfully")
            # Add a delay after jupyter completion
            time.sleep(5)  # 5 second delay
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running jupyter: {e.stderr}")
            return False
            
    def run_convertv2(self) -> bool:
        """Run convertv2 script."""
        try:
            logging.info("Running convertv2")
            # Add a delay before starting convert
            time.sleep(3)  # 3 second delay
            result = subprocess.run(
                ['python', 'convertv2.py', '--input', 'all', '--output', 'testt/yes'],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info("Convertv2 completed successfully")
            # Add a delay after convert completion
            time.sleep(3)  # 3 second delay
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running convertv2: {e.stderr}")
            return False
            
    def check_file_exists(self, filepath: str, timeout: int = 30) -> bool:
        """Check if a file exists with timeout and proper error handling."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if os.path.exists(filepath):
                    # Add a small delay to ensure file is fully written
                    time.sleep(1)
                    return True
            except Exception as e:
                logging.error(f"Error checking file {filepath}: {str(e)}")
            time.sleep(1)
        return False
            
    def wait_for_jupyter_output(self, address: str) -> bool:
        """Wait for jupyter to create output file."""
        expected_file = os.path.join(self.all_dir, f'transactions_{address}.json')
        logging.info(f"Waiting for jupyter output file: {expected_file}")
        
        if self.check_file_exists(expected_file):
            logging.info(f"Found jupyter output file: {expected_file}")
            time.sleep(2)  # Additional delay after finding file
            return True
            
        logging.error(f"Timeout waiting for jupyter output: {expected_file}")
        return False
        
    def wait_for_convert_output(self, address: str) -> Tuple[Optional[str], Optional[str]]:
        """Wait for convertv2 to create output files."""
        logging.info(f"Waiting for convert output files for address: {address}")
        time.sleep(2)  # Initial delay before checking
        
        # Look for both enhanced and price history files
        enhanced_pattern = os.path.join(self.testt_yes_dir, f'enhanced_{address}_*.json')
        price_pattern = os.path.join(self.testt_yes_dir, f'price_history_{address}_*.json')
        
        enhanced_files = glob.glob(enhanced_pattern)
        price_files = glob.glob(price_pattern)
        
        if enhanced_files and price_files:
            time.sleep(2)  # Delay after finding files
            return enhanced_files[0], price_files[0]
        
        logging.error(f"Could not find convert output files for address: {address}")
        return None, None
        
    def process_address(self, address: str) -> bool:
        """Process a single address through the pipeline."""
        try:
            logging.info(f"\nProcessing address: {address}")
            
            # Run jupyter and wait for output
            if not self.run_jupyter(address):
                return False
                
            if not self.wait_for_jupyter_output(address):
                return False
                
            # Run convertv2
            if not self.run_convertv2():
                return False
                
            # Wait for convert output
            enhanced_file, price_file = self.wait_for_convert_output(address)
            if not enhanced_file or not price_file:
                return False
                
            # Copy files to newyesV2
            try:
                shutil.copy2(enhanced_file, self.newyesv2_dir)
                shutil.copy2(price_file, self.newyesv2_dir)
                logging.info(f"Copied files to {self.newyesv2_dir}")
                
                # Delete transaction file from all directory
                transaction_file = os.path.join(self.all_dir, f'transactions_{address}.json')
                if os.path.exists(transaction_file):
                    os.remove(transaction_file)
                    logging.info(f"Deleted {transaction_file}")
                
                time.sleep(2)  # Delay after completing process
                return True
                
            except Exception as e:
                logging.error(f"Error copying/deleting files: {str(e)}")
                return False
                
        except Exception as e:
            logging.error(f"Error processing address {address}: {str(e)}")
            return False
            
    def run(self):
        """Run the complete cleaning process."""
        try:
            logging.info("Starting data cleaning process")
            
            # Get all enhanced files
            enhanced_files = self.get_enhanced_files()
            logging.info(f"Found {len(enhanced_files)} enhanced files to process")
            
            processed = 0
            failed = 0
            
            for enhanced_file in enhanced_files:
                try:
                    address = self.extract_address_from_filename(os.path.basename(enhanced_file))
                    if not address:
                        logging.error(f"Could not extract address from {enhanced_file}")
                        failed += 1
                        continue
                        
                    if self.process_address(address):
                        processed += 1
                        time.sleep(3)  # Delay between processing addresses
                    else:
                        failed += 1
                        
                except KeyboardInterrupt:
                    logging.info("\nProcess interrupted by user")
                    break
                except Exception as e:
                    logging.error(f"Error processing file {enhanced_file}: {str(e)}")
                    failed += 1
                    
            logging.info(f"\nProcessing complete:")
            logging.info(f"Successfully processed: {processed}")
            logging.info(f"Failed: {failed}")
            
        except KeyboardInterrupt:
            logging.info("\nProcess interrupted by user")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Fatal error in main process: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    cleaner = DataCleaner()
    try:
        cleaner.run()
    except KeyboardInterrupt:
        logging.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)