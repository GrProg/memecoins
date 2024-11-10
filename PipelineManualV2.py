import os
import sys
import shutil
import glob
import subprocess
import time
from datetime import datetime
import re
from typing import Tuple, Optional

class PipelineManualV3:
    def __init__(self, coin_address: str):
        self.coin_address = coin_address
        self.project_dir = os.getcwd()
        self.all_dir = os.path.join(self.project_dir, 'all')
        self.pameligo_dir = os.path.join(self.project_dir, 'pameligo')
        self.pred_yes_dir = os.path.join(self.pameligo_dir, 'PredYes')
        self.model_path = os.path.join(self.pameligo_dir, "mcap_prediction_model.joblib")  # Updated model path
        
        # Ensure required directories exist
        os.makedirs(self.all_dir, exist_ok=True)
        os.makedirs(self.pred_yes_dir, exist_ok=True)

    def check_model_exists(self) -> bool:
        """Check if the required model file exists"""
        if not os.path.exists(self.model_path):
            print("\nError: Required model file not found!")
            print(f"Expected model file: {self.model_path}")
            print("\nPlease ensure the model file exists in the pameligo directory")
            return False
        return True

    def run_jupyter(self) -> Optional[str]:
        """Step 1: Run jupyter.py and handle the output file"""
        print("\nStep 1: Running jupyter.py...")
        
        # Run jupyter.py with coin address
        subprocess.run([sys.executable, 'jupyter.py', self.coin_address])
        
        # Find the generated file
        files = glob.glob(f'all/transactions_{self.coin_address}.json')
        if not files:
            raise Exception("Jupyter.py didn't generate any output file")
            
        print(f"✓ Successfully generated transaction data")
        return files[0]

    def run_convert_v2(self) -> Tuple[str, str]:
        """Step 2: Run convertv2.py"""
        print("\nStep 2: Running convertv2.py...")
        subprocess.run([sys.executable, 'convertv2.py', '--input', 'all', '--output', 'pameligo/PredYes'])
        
        # Wait a moment for files to be created
        time.sleep(2)
        
        # Find the generated files
        enhanced_files = glob.glob(os.path.join(self.pred_yes_dir, f'enhanced_{self.coin_address}_*.json'))
        price_files = glob.glob(os.path.join(self.pred_yes_dir, f'price_history_{self.coin_address}_*.json'))
        
        if not enhanced_files or not price_files:
            raise Exception("Convertv2.py didn't generate the expected output files")
            
        # Sort files by modification time to get the most recent ones
        enhanced_file = sorted(enhanced_files, key=os.path.getmtime)[-1]
        price_file = sorted(price_files, key=os.path.getmtime)[-1]
        
        # Extract market cap from filename for logging
        mcap_match = re.search(r'_(\d+)\.json$', enhanced_file)
        mcap = mcap_match.group(1) if mcap_match else "unknown"
        
        print(f"✓ Successfully generated enhanced data and price history (Max MCap: ${mcap})")
        return enhanced_file, price_file

    def run_predictor_v3(self, enhanced_file: str, price_file: str):
        """Step 3: Run predictorV3.py"""
        print("\nStep 3: Running predictorV3.py...")
        
        if not self.check_model_exists():
            raise Exception("Required model file not found")
        
        # Extract just the filenames for display
        enhanced_name = os.path.basename(enhanced_file)
        price_name = os.path.basename(price_file)
        
        print(f"Processing files:")
        print(f"Enhanced: {enhanced_name}")
        print(f"Price: {price_name}")
        
        # Change to pameligo directory before running predictor
        original_dir = os.getcwd()
        os.chdir(self.pameligo_dir)
        
        try:
            # Create and run the prediction code
            predict_code = f"""
import sys
from market_predictor import predict_market_cap

price_file = 'PredYes/{price_name}'
enhanced_file = 'PredYes/{enhanced_name}'

result = predict_market_cap(price_file, enhanced_file)
"""
            # Save to temporary file
            with open('temp_predict.py', 'w') as f:
                f.write(predict_code)
            
            # Run prediction
            result = subprocess.run(
                [sys.executable, 'temp_predict.py'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("Warning: Predictor encountered errors:")
                print(result.stderr)
            else:
                print(result.stdout)
                
            print(f"✓ Completed prediction analysis")
            
        finally:
            # Clean up temp file
            try:
                os.remove('temp_predict.py')
            except:
                pass
            # Change back to original directory
            os.chdir(original_dir)

    def run_pipeline(self):
        """Run the complete pipeline"""
        try:
            print(f"Starting pipeline for coin address: {self.coin_address}")
            
            # Step 1: Run jupyter and get the transaction file
            transaction_file = self.run_jupyter()
            
            # Step 2: Run convertv2 and get the generated files
            enhanced_file, price_file = self.run_convert_v2()
            
            # Step 3: Run predictorV3
            self.run_predictor_v3(enhanced_file, price_file)
            
            print("\nPipeline completed successfully!")
            
        except Exception as e:
            print(f"\nError in pipeline: {str(e)}")
            sys.exit(1)
        finally:
            # Clean up transaction file from 'all' directory
            try:
                for f in glob.glob(os.path.join(self.all_dir, f'transactions_{self.coin_address}*.json')):
                    os.remove(f)
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {str(e)}")
    def run_predictor_v3(self, enhanced_file: str, price_file: str):
        """Step 3: Run PredictorV3.py"""
        print("\nStep 3: Running PredictorV3.py...")
        
        if not self.check_model_exists():
            raise Exception("Required model file not found")
        
        # Extract just the filenames for display
        enhanced_name = os.path.basename(enhanced_file)
        price_name = os.path.basename(price_file)
        
        print(f"Processing files:")
        print(f"Enhanced: {enhanced_name}")
        print(f"Price: {price_name}")
        
        # Change to pameligo directory before running predictor
        original_dir = os.getcwd()
        os.chdir(self.pameligo_dir)
        
        try:
            # Run prediction using PredictorV3.py directly
            result = subprocess.run(
                [sys.executable, 'PredictorV3.py', 
                 os.path.join('PredYes', enhanced_name), 
                 os.path.join('PredYes', price_name)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("Warning: Predictor encountered errors:")
                print(result.stderr)
            else:
                print(result.stdout)
                
            print(f"✓ Completed prediction analysis")
            
        finally:
            # Change back to original directory
            os.chdir(original_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python PipelineManualV3.py [coin address]")
        sys.exit(1)
        
    pipeline = PipelineManualV3(sys.argv[1])
    pipeline.run_pipeline()