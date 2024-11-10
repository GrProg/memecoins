import requests
import json

def setup_helius_webhook(token_address: str):
    """Set up Helius webhook for a token address"""
    webhook_url = "https://your-ngrok-url.ngrok-free.app/webhook"  # Update this
    api_key = "30b8e7e2-9206-41ca-a392-112845774aef"
    
    try:
        # Delete existing webhooks first
        list_url = f"https://api.helius.xyz/v0/webhooks?api-key={api_key}"
        existing_webhooks = requests.get(list_url).json()
        
        for webhook in existing_webhooks:
            webhook_id = webhook.get('webhookID')
            if webhook_id:
                delete_url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={api_key}"
                requests.delete(delete_url)
                print(f"Deleted webhook: {webhook_id}")
        
        # Create new webhook with enhanced data
        url = f"https://api.helius.xyz/v0/webhooks?api-key={api_key}"
        
        webhook_config = {
            "webhookURL": webhook_url,
            "transactionTypes": ["ANY"],
            "accountAddresses": [token_address],
            "webhookType": "enhanced"  # This is important
        }
        
        response = requests.post(url, json=webhook_config)
        response.raise_for_status()
        
        print("\nWebhook successfully registered!")
        print(json.dumps(response.json(), indent=2))
        
        return response.json()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
if __name__ == "__main__":
    #token_address = input("Enter token address to monitor: ")
    setup_helius_webhook("7wbiW1XuGD77F4tSDWyMu73bJKwWX4zKwDfjdPRMpump")