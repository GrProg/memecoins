from flask import Flask, request, jsonify
import json
import os
from datetime import datetime
import logging
import traceback
from typing import Dict, List, Optional

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webhook.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TransactionManager:
    def __init__(self, storage_dir: str = 'all'):
        """Initialize transaction manager with storage directory"""
        self.storage_dir = storage_dir
        self.ensure_storage_dir()
        self.transaction_count = 0
        logger.info(f"Initialized TransactionManager with storage directory: {storage_dir}")

    def ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            logger.info(f"Storage directory confirmed: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Error creating storage directory: {str(e)}")
            raise

    def get_transaction_file(self, token_address: str) -> str:
        """Get the full path for a token's transaction file"""
        return os.path.join(self.storage_dir, f'transactions_{token_address}.json')

    def load_existing_transactions(self, filename: str) -> List[Dict]:
        """Load existing transactions from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in file: {filename}")
                return []
            except Exception as e:
                logger.error(f"Error loading transactions: {str(e)}")
                return []
        return []

    def is_duplicate_transaction(self, tx_signature: str, existing_transactions: List[Dict]) -> bool:
        """Check if transaction already exists"""
        return any(tx.get('signature') == tx_signature for tx in existing_transactions)

    def save_transaction(self, token_address: str, transaction: Dict) -> bool:
        """Save a new transaction to storage"""
        if not token_address or not transaction:
            logger.error("Invalid token address or transaction data")
            return False

        try:
            filename = self.get_transaction_file(token_address)
            transactions = self.load_existing_transactions(filename)
            
            tx_signature = transaction.get('signature')
            if not tx_signature:
                logger.warning("Transaction missing signature")
                return False

            if self.is_duplicate_transaction(tx_signature, transactions):
                logger.info(f"Duplicate transaction ignored: {tx_signature}")
                return False

            # Add new transaction
            transactions.append(transaction)
            
            # Save updated file
            with open(filename, 'w') as f:
                json.dump(transactions, f, indent=2)

            self.transaction_count += 1
            logger.info(f"Saved new transaction for {token_address}. Total: {self.transaction_count}")
            return True

        except Exception as e:
            logger.error(f"Error saving transaction: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        try:
            files = os.listdir(self.storage_dir)
            transaction_files = [f for f in files if f.startswith('transactions_')]
            
            stats = {
                'total_tokens': len(transaction_files),
                'total_transactions': self.transaction_count,
                'storage_size': sum(os.path.getsize(os.path.join(self.storage_dir, f)) 
                                  for f in transaction_files),
                'last_update': datetime.now().isoformat()
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}

# Initialize Flask app and transaction manager
app = Flask(__name__)
tx_manager = TransactionManager()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for basic verification"""
    try:
        stats = tx_manager.get_statistics()
        return jsonify({
            "status": "running",
            "time": datetime.now().isoformat(),
            "storage_dir": tx_manager.storage_dir,
            "statistics": stats,
            "endpoints": ["/", "/health", "/webhook"]
        })
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        storage_exists = os.path.exists(tx_manager.storage_dir)
        storage_writable = os.access(tx_manager.storage_dir, os.W_OK)
        
        health_status = {
            "status": "healthy" if (storage_exists and storage_writable) else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "storage": {
                "directory": tx_manager.storage_dir,
                "exists": storage_exists,
                "writable": storage_writable
            },
            "transactions_processed": tx_manager.transaction_count
        }
        
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Main webhook endpoint for receiving transaction data"""
    TARGET_TOKEN = "Gh44uRQQSpKrYpgufCwPKVs3MYzHwV6APHzsotVMpump"
    
    logger.info("Received webhook request")
    
    try:
        # Get the data and log it
        data = request.get_json()
        if not data:
            logger.error("No data received")
            return jsonify({"status": "error", "message": "No data received"}), 400

        # Find all transactions in this block of data
        transactions = data if isinstance(data, list) else [data]
        logger.info(f"Processing {len(transactions)} transactions")
        
        processed_count = 0
        saved_count = 0
        
        for tx in transactions:
            processed_count += 1
            
            # Only process if it matches expected format
            if all(key in tx for key in ['description', 'type', 'tokenTransfers', 'signature']):
                token_transfers = tx.get('tokenTransfers', [])
                
                # Check if any transfer involves our target token
                for transfer in token_transfers:
                    if transfer.get('mint') == TARGET_TOKEN:
                        # Print transaction details immediately
                        print("\n" + "="*50)
                        print(f"🔔 NEW TRANSACTION DETECTED!")
                        print(f"Type: {tx.get('type')}")
                        print(f"Time: {datetime.fromtimestamp(tx.get('timestamp', 0)).strftime('%H:%M:%S')}")
                        print(f"Amount: {transfer.get('tokenAmount', 'Unknown')}")
                        
                        # If there's a price, show it
                        if 'nativeTransfers' in tx:
                            sol_amount = sum(nt['amount'] for nt in tx['nativeTransfers']) / 1e9
                            print(f"SOL Amount: {sol_amount:.4f} SOL")
                            
                            if transfer.get('tokenAmount'):
                                token_amount = float(transfer.get('tokenAmount', 0))
                                if token_amount > 0:
                                    price = sol_amount / token_amount
                                    print(f"Price: {price:.8f} SOL/token")
                        
                        print(f"Signature: {tx['signature'][:16]}...")
                        print("="*50)
                        
                        # Save transaction
                        if tx_manager.save_transaction(TARGET_TOKEN, tx):
                            saved_count += 1
                            logger.info(f"Saved transaction {tx['signature'][:8]}")
                        break

        result = {
            "status": "success",
            "processed": processed_count,
            "saved": saved_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save debug data
        debug_dir = 'debug'
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f'webhook_{datetime.now().strftime("%H%M%S")}.json')
        with open(debug_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = f"Error processing webhook: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": error_msg}), 500
    
@app.route('/webhook/test', methods=['GET'])
def test_webhook():
    """Test endpoint"""
    return jsonify({
        "status": "webhook endpoint working",
        "time": datetime.now().isoformat(),
        "target_token": "Gh44uRQQSpKrYpgufCwPKVs3MYzHwV6APHzsotVMpump"
    })

@app.errorhandler(Exception)
def handle_error(e):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({"status": "error", "message": str(e)}), 500

def start_server(host: str = '0.0.0.0', port: int = 5000):
    """Start the Flask server"""
    logger.info(f"Starting webhook server on {host}:{port}...")
    try:
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)