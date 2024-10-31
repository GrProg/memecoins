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
import websockets
import json
from datetime import datetime
import sys

class PumpMonitor:
    def __init__(self, token_address):
        self.token_address = token_address
        self.ws_url = "wss://pumpportal.fun/api/data"
        self.target_mcap = 12000  # 98k target
        
    async def connect(self):
        """Connect to PumpFun websocket"""
        async with websockets.connect(self.ws_url) as websocket:
            print(f"Connected to PumpFun API - Monitoring {self.token_address}")
            
            # Subscribe to trades for our specific token
            await websocket.send(json.dumps({
                "method": "subscribeTokenTrade",
                "keys": [self.token_address]
            }))
            
            while True:
                try:
                    message = await websocket.recv()
                    await self.process_update(json.loads(message))
                except Exception as e:
                    print(f"Error: {e}")
                    await asyncio.sleep(5)  # Wait before retry
    
    async def process_update(self, data):
        """Process market cap updates"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log update
        with open(f'mcap_updates_{self.token_address}.txt', 'a') as f:
            f.write(f"\n=== Update at {timestamp} ===\n")
            f.write(json.dumps(data, indent=2))
            f.write("\n" + "="*50 + "\n")
        
        # Check if we have market cap info
        if 'marketCap' in data:
            mcap = int(data['marketCap'])
            print(f"\nMarket Cap Update: ${mcap:,}")
            
            # Check if we hit our target
            if mcap > self.target_mcap:
                print(f"🎯 Target reached! Market Cap: ${mcap:,}")
                await self.execute_sell()
                sys.exit(0)  # Stop monitoring after sell
    
    async def execute_sell(self):
        """Execute sell when target is hit"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the sell signal
        with open('trades.txt', 'a') as f:
            f.write(f"\n=== SELL SIGNAL at {timestamp} ===\n")
            f.write(f"Token: {self.token_address}\n")
            f.write(f"Reason: Market cap exceeded 98k\n")
            f.write("="*50 + "\n")
        
        print(f"Sell signal generated for {self.token_address}")

async def main():
    if len(sys.argv) != 2:
        print("Please provide token address")
        return
        
    token_address = sys.argv[1]
    monitor = PumpMonitor(token_address)
    await monitor.connect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error: {e}")