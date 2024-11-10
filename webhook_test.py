# test_local.py
import requests

def test_local_server():
    """Test if webhook server is running locally"""
    endpoints = ['/', '/health', '/webhook']
    base_url = 'http://localhost:5000'
    
    print("Testing local webhook server...")
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            if endpoint == '/webhook':
                # Webhook endpoint expects POST
                response = requests.post(url, json={})
            else:
                response = requests.get(url)
            
            print(f"\nEndpoint: {endpoint}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
        except requests.exceptions.ConnectionError:
            print(f"\nError: Could not connect to {url}")
            print("Is the server running?")
            return False
    return True

if __name__ == "__main__":
    test_local_server()