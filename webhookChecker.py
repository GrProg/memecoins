import requests

def update_webhook():
    API_KEY = "30b8e7e2-9206-41ca-a392-112845774aef"
    WEBHOOK_ID = "c772df10-488a-44d0-ba63-a610eaeb4dec"
    NEW_URL = input("Enter your new ngrok URL (e.g., https://xxxx-xx-xxx-xx-xx.ngrok-free.app/webhook): ")
    
    # Update webhook
    url = f"https://api.helius.xyz/v0/webhooks/{WEBHOOK_ID}?api-key={API_KEY}"
    
    data = {
        "webhookURL": NEW_URL,
        "transactionTypes": ["ANY"],
        "accountAddresses": ["HD9iNo8TAcAVuCzizj9W6XCotEyRiP7jaR3NxL4Hpump"],
        "webhookType": "enhanced"
    }
    
    try:
        response = requests.put(url, json=data)
        response.raise_for_status()
        print(f"\nSuccessfully updated webhook URL to: {NEW_URL}")
        print("\nNew webhook configuration:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error updating webhook: {str(e)}")

if __name__ == "__main__":
    update_webhook()