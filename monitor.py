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
import time
from datetime import datetime
import os

class PumpPortalDetector:
    def __init__(self):
        self.ws_uri = "wss://pumpportal.fun/api/data"
        self.processed_tokens = set()
        self.active_tokens = {}
        self.SOL_PRICE = 170  # Updated SOL price
        self.data_directory = "pump_data"
        
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
    def save_to_file(self, event_type, data):
        try:
            filename = os.path.join(self.data_directory, f'pump_data_{datetime.now().strftime("%Y%m%d")}.txt')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_data = f"[{timestamp}] {event_type}: {json.dumps(data)}\n"
            
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(formatted_data)
            
            print(f"Saved {event_type} to {filename}")
            
        except Exception as e:
            print(f"Error saving to file: {str(e)}")
        
    async def monitor_new_tokens(self):
        while True:
            try:
                async with websockets.connect(self.ws_uri) as websocket:
                    print(f"Connected to websocket. Data will be saved to {self.data_directory}/")
                    print(f"Using SOL price of ${self.SOL_PRICE} USD")
                    
                    await websocket.send(json.dumps({
                        "method": "subscribeNewToken"
                    }))
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            # New token detection
                            if "mint" in data and "marketCapSol" in data:
                                token_mint = data["mint"]
                                if token_mint in self.processed_tokens:
                                    continue
                                    
                                self.processed_tokens.add(token_mint)
                                mcap = float(data["marketCapSol"]) * self.SOL_PRICE
                                
                                print(f"\nNew token: {token_mint}")
                                print(f"MarketCap SOL: {data['marketCapSol']}")
                                print(f"MarketCap USD: ${mcap:,.2f}")
                                
                                # Save token creation data
                                creation_data = {
                                    "token": token_mint,
                                    "mcap_sol": float(data["marketCapSol"]),
                                    "mcap_usd": mcap,
                                    "raw_data": data
                                }
                                self.save_to_file("TOKEN_CREATED", creation_data)
                                
                                await websocket.send(json.dumps({
                                    "method": "subscribeTokenTrade",
                                    "keys": [token_mint]
                                }))
                                
                                self.active_tokens[token_mint] = {
                                    'discovery_time': time.time(),
                                    'initial_mcap': mcap,
                                    'peak_mcap': mcap,
                                    'trades': []
                                }
                                
                                # Check for initial spike
                                # 15k to 30k range
                                if 15000 <= mcap <= 30000:
                                    spike_data = {
                                        "token": token_mint,
                                        "mcap_sol": float(data["marketCapSol"]),
                                        "mcap_usd": mcap,
                                        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        "raw_data": data
                                    }
                                    print(f"\nSPIKE DETECTED!")
                                    print(f"Token: {token_mint}")
                                    print(f"MCap SOL: {data['marketCapSol']}")
                                    print(f"MCap USD: ${mcap:,.2f}")
                                    self.save_to_file("SPIKE_DETECTED", spike_data)
                            
                            # Trade processing
                            elif "trade" in data:
                                trade_data = data["trade"]
                                token_mint = trade_data.get("mint")
                                
                                if token_mint in self.active_tokens:
                                    token = self.active_tokens[token_mint]
                                    mcap_sol = float(trade_data.get("marketCapSol", 0))
                                    mcap = mcap_sol * self.SOL_PRICE
                                    
                                    # Save significant trades
                                    if mcap >= 15000:
                                        trade_info = {
                                            "token": token_mint,
                                            "mcap_sol": mcap_sol,
                                            "mcap_usd": mcap,
                                            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            "raw_data": trade_data
                                        }
                                        self.save_to_file("SIGNIFICANT_TRADE", trade_info)
                                    
                                    if mcap > token['peak_mcap']:
                                        token['peak_mcap'] = mcap
                                        peak_data = {
                                            "token": token_mint,
                                            "mcap_sol": mcap_sol,
                                            "mcap_usd": mcap,
                                            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        }
                                        self.save_to_file("NEW_PEAK", peak_data)
                                    
                                    # Check for drop
                                    if token['peak_mcap'] > 15000:
                                        drop = ((mcap - token['peak_mcap']) / token['peak_mcap']) * 100
                                        if drop <= -50:
                                            drop_data = {
                                                "token": token_mint,
                                                "initial_mcap_usd": token['initial_mcap'],
                                                "peak_mcap_usd": token['peak_mcap'],
                                                "final_mcap_usd": mcap,
                                                "drop_percent": drop,
                                                "time_elapsed": time.time() - token['discovery_time']
                                            }
                                            print(f"\nDROP DETECTED!")
                                            print(f"Token: {token_mint}")
                                            print(f"From Peak: ${token['peak_mcap']:,.2f}")
                                            print(f"To: ${mcap:,.2f}")
                                            print(f"Drop: {drop:.2f}%")
                                            self.save_to_file("DROP_DETECTED", drop_data)
                                            del self.active_tokens[token_mint]
                            
                        except Exception as e:
                            print(f"Error processing message: {str(e)}")
                            self.save_to_file("ERROR", {"error": str(e), "message": str(message)})
                            
            except Exception as e:
                print(f"Websocket connection error: {str(e)}")
                print("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

async def main():
    detector = PumpPortalDetector()
    await detector.monitor_new_tokens()

if __name__ == "__main__":
    asyncio.run(main())
