# test_trade.py
from telethon import TelegramClient
import asyncio
from monitor import execute_trade  # Import the function we just created

# Use the same credentials as in monitor.py
API_ID = 21528285
API_HASH = '17502cc14c93332c2537b7838d6db5bf'
PHONE = '+306988738227'

async def test_trade():
    print("Starting test trade...")
    
    # Initialize the client
    client = TelegramClient('session_name', API_ID, API_HASH)
    await client.start(phone=PHONE)
    
    # Test token address - replace this with the address you want to test
    test_token = "YOUR_TEST_TOKEN_ADDRESS"
    
    try:
        # Execute the trade
        success = await execute_trade(client, test_token)
        
        if success:
            print("Test trade executed successfully!")
        else:
            print("Test trade failed")
            
    except Exception as e:
        print(f"Error during test: {e}")
    
    finally:
        await client.disconnect()

if __name__ == '__main__':
    asyncio.run(test_trade())