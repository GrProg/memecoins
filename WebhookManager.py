import requests
import json
import logging
from datetime import datetime
import subprocess
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webhook_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WebhookManager:
    def __init__(self):
        self.API_KEY = "30b8e7e2-9206-41ca-a392-112845774aef"
        self.TOKEN_ADDRESS = "HD9iNo8TAcAVuCzizj9W6XCotEyRiP7jaR3NxL4Hpump"
        self.current_webhook_id = None
        
    def check_flask_server(self) -> bool:
        """Check if Flask server is running"""
        try:
            response = requests.get("http://localhost:5000/health")
            return response.status_code == 200
        except:
            return False
            
    def get_ngrok_url(self) -> str:
        """Get current ngrok URL"""
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()['tunnels']
            if tunnels:
                return tunnels[0]['public_url']
        except:
            pass
        return None

    def list_webhooks(self):
        """List all registered webhooks"""
        try:
            url = f"https://api.helius.xyz/v0/webhooks?api-key={self.API_KEY}"
            response = requests.get(url)
            webhooks = response.json()
            
            print("\nCurrent Webhooks:")
            print("================")
            for webhook in webhooks:
                self.current_webhook_id = webhook['webhookID']  # Save the ID
                print(f"ID: {webhook['webhookID']}")
                print(f"URL: {webhook['webhookURL']}")
                print(f"Addresses: {webhook['accountAddresses']}")
                print(f"Type: {webhook['webhookType']}")
                print("-" * 50)
                
            return webhooks
        except Exception as e:
            print(f"Error listing webhooks: {str(e)}")
            return []

    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a specific webhook"""
        try:
            url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={self.API_KEY}"
            response = requests.delete(url)
            return response.status_code == 200
        except Exception as e:
            print(f"Error deleting webhook: {str(e)}")
            return False

    def create_webhook(self, webhook_url: str) -> bool:
        """Create a new webhook"""
        try:
            url = f"https://api.helius.xyz/v0/webhooks?api-key={self.API_KEY}"
            data = {
                "webhookURL": webhook_url,
                "transactionTypes": ["ANY"],
                "accountAddresses": [self.TOKEN_ADDRESS],
                "webhookType": "enhanced"
            }
            response = requests.post(url, json=data)
            if response.status_code == 200:
                webhook_data = response.json()
                self.current_webhook_id = webhook_data['webhookID']
                print(f"\nWebhook created successfully!")
                print(f"ID: {webhook_data['webhookID']}")
                return True
            return False
        except Exception as e:
            print(f"Error creating webhook: {str(e)}")
            return False

    def update_webhook_url(self, new_url: str = None):
        """Update webhook with new URL"""
        if not new_url:
            new_url = self.get_ngrok_url()
            if not new_url:
                new_url = input("\nEnter your new ngrok URL: ")
        
        # Ensure URL ends with /webhook
        if not new_url.endswith('/webhook'):
            new_url = f"{new_url}/webhook"
            
        # First, list and delete existing webhooks
        existing_webhooks = self.list_webhooks()
        for webhook in existing_webhooks:
            self.delete_webhook(webhook['webhookID'])
            
        # Create new webhook
        success = self.create_webhook(new_url)
        if success:
            print(f"\nWebhook URL updated to: {new_url}")
        else:
            print("\nFailed to update webhook URL")

    def test_webhook(self):
        """Test webhook connection"""
        webhooks = self.list_webhooks()
        if not webhooks:
            print("No webhooks found!")
            return
            
        webhook = webhooks[0]  # Test first webhook
        webhook_url = webhook['webhookURL']
        print(f"\nTesting webhook URL: {webhook_url}")
        
        # Get base URL
        base_url = webhook_url.rsplit('/webhook', 1)[0]
        
        # First check if Flask server is running
        if not self.check_flask_server():
            print("❌ Flask server is not running! Start it with: python webhook.py")
            return
            
        # Check if ngrok is running
        ngrok_url = self.get_ngrok_url()
        if not ngrok_url:
            print("❌ ngrok is not running! Start it with: ngrok http 5000")
            return
            
        # Test endpoints
        endpoints = ['/', '/health', '/webhook']
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            print(f"\nTesting {url}")
            try:
                if endpoint == '/webhook':
                    response = requests.post(url, json={'test': True})
                else:
                    response = requests.get(url)
                    
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    print("✅ Endpoint working")
                else:
                    print("❌ Endpoint returned error")
            except requests.exceptions.RequestException as e:
                print(f"❌ Connection error: {str(e)}")

def main():
    manager = WebhookManager()
    
    while True:
        print("\nWebhook Manager")
        print("==============")
        print("1. List current webhooks")
        print("2. Update webhook URL")
        print("3. Test webhook connection")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            manager.list_webhooks()
        elif choice == '2':
            manager.update_webhook_url()
        elif choice == '3':
            manager.test_webhook()
        elif choice == '4':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()