#USE ONLY FOR LIVE DATA
import os
import sys
import shutil
import glob
import subprocess
import time
from datetime import datetime
import re

class NewPipeline:
    def __init__(self, coin_address):
        self.coin_address = coin_address
        self.project_dir = os.getcwd()
        self.all_dir = os.path.join(self.project_dir, 'all')
        self.testt_dir = os.path.join(self.project_dir, 'testt')
        self.testt_yes_dir = os.path.join(self.testt_dir, 'yes')
        self.pameligo_dir = os.path.join(self.project_dir, 'pameligo')
        
        # Ensure required directories exist
        os.makedirs(self.all_dir, exist_ok=True)
        os.makedirs(self.testt_yes_dir, exist_ok=True)
        os.makedirs(self.pameligo_dir, exist_ok=True)

    def run_jupyter(self):
        """Step 1: Run jupyter.py and handle the output file"""
        print("\nStep 1: Running jupyter.py...")
        
        # Run jupyter.py with coin address
        subprocess.run([sys.executable, 'jupyter.py', self.coin_address])
        
        # Find the generated file
        files = glob.glob(f'all_transactions_{self.coin_address[:8]}*.json')
        if not files:
            raise Exception("Jupyter.py didn't generate any output file")
            
        original_file = files[0]
        
        # Create new filename
        timestamp = re.search(r'_(\d{8}_\d{6})\.json$', original_file).group(1)
        new_filename = f"transactions_{self.coin_address}_{timestamp}.json"
        
        # Rename and move to all directory
        shutil.move(original_file, os.path.join(self.all_dir, new_filename))
        print(f"Moved and renamed file to: {new_filename}")
        return new_filename

    def run_convert(self):
        """Step 2: Run convert.py"""
        print("\nStep 2: Running convert.py...")
        subprocess.run([sys.executable, 'convert.py', '--input', 'all', '--output', 'testt/yes'])
        
        # Wait a moment for files to be created
        time.sleep(2)
        
        # Find the generated files
        enhanced_files = glob.glob(os.path.join(self.testt_yes_dir, f'enhanced_{self.coin_address}_*.json'))
        price_files = glob.glob(os.path.join(self.testt_yes_dir, f'price_history_{self.coin_address}_*.json'))
        
        if not enhanced_files or not price_files:
            raise Exception("Convert.py didn't generate the expected output files")
            
        return enhanced_files[0], price_files[0]

    def run_json_cleaner(self):
        """Step 3: Run JsonCleaner.py"""
        print("\nStep 3: Running JsonCleaner.py...")
        os.chdir(self.testt_dir)
        subprocess.run([sys.executable, 'JsonCleaner.py'])
        os.chdir(self.project_dir)

    def move_files_to_pameligo(self, enhanced_file, price_file):
        """Step 4: Move files to pameligo directory"""
        print("\nStep 4: Moving files to pameligo directory...")
        
        # Get just the filenames
        enhanced_filename = os.path.basename(enhanced_file)
        price_filename = os.path.basename(price_file)
        
        # Move files to pameligo
        shutil.move(enhanced_file, os.path.join(self.pameligo_dir, enhanced_filename))
        shutil.move(price_file, os.path.join(self.pameligo_dir, price_filename))
        
        # Clean up the original file from 'all' directory
        all_files = glob.glob(os.path.join(self.all_dir, f'transactions_{self.coin_address}_*.json'))
        for file in all_files:
            os.remove(file)
            
        return enhanced_filename, price_filename

    def run_predictor(self, enhanced_filename, price_filename):
        """Step 5: Run predictor.py"""
        print("\nStep 5: Running predictor.py...")
        os.chdir(self.pameligo_dir)
        subprocess.run([sys.executable, 'predictor.py', enhanced_filename, price_filename])
        os.chdir(self.project_dir)

    def run_pipeline(self):
        """Run the complete pipeline"""
        try:
            print(f"Starting pipeline for coin address: {self.coin_address}")
            
            # Step 1: Run jupyter and get the transaction file
            transaction_file = self.run_jupyter()
            
            # Step 2: Run convert and get the generated files
            enhanced_file, price_file = self.run_convert()
            
            # Step 3: Run JsonCleaner
            self.run_json_cleaner()
            
            # Step 4: Move files to pameligo
            enhanced_filename, price_filename = self.move_files_to_pameligo(enhanced_file, price_file)
            
            # Step 5: Run predictor
            self.run_predictor(enhanced_filename, price_filename)
            
            print("\nPipeline completed successfully!")
            
        except Exception as e:
            print(f"\nError in pipeline: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python NewPipeline.py [coin address]")
        sys.exit(1)
        
    pipeline = NewPipeline(sys.argv[1])
    pipeline.run_pipeline()