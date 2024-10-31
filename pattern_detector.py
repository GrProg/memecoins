import asyncio
import websockets
import json
import time
from datetime import datetime
import sys
import io

class Logger:
    def __init__(self):
        self.log_file = f'pump_detector_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
    def log(self, message):
        #print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

class PumpPortalDetector:
    def __init__(self):
        self.ws_uri = "wss://pumpportal.fun/api/data"
        self.saved_tokens = set()
        self.active_tokens = {}
        self.SOL_PRICE = 170
        self.logger = Logger()
        
    async def monitor_new_tokens(self):
        async with websockets.connect(self.ws_uri) as websocket:
            self.logger.log("Started monitoring for new tokens...")
            
            await websocket.send(json.dumps({
                "method": "subscribeNewToken"
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "mint" in data:
                        token_mint = data["mint"]
                        if token_mint in self.saved_tokens:
                            continue
                            
                        # Try to get market cap from the data
                        mcap = 0
                        if "marketCap" in data:
                            mcap = float(data["marketCap"])
                        elif "market_cap" in data:
                            mcap = float(data["market_cap"])
                        elif "marketCapSol" in data:
                            mcap = float(data["marketCapSol"]) * self.SOL_PRICE
                            
                        self.logger.log(f"\nNew token detected: {token_mint} with initial mcap: ${mcap:,.2f}")
                        
                        # Save immediately if mcap is already >= 15000
                        if mcap >= 15000:
                            pump_msg = f"INSTANT PUMP DETECTED!\nToken: {token_mint}\nInitial MCap: ${mcap:,.2f}"
                            self.logger.log(pump_msg)
                            await self.save_pump(token_mint, mcap, 0, mcap)
                            self.saved_tokens.add(token_mint)
                            continue
                        
                        # Otherwise track the token
                        self.active_tokens[token_mint] = {
                            'start_time': time.time(),
                            'initial_mcap': mcap,
                            'highest_mcap': mcap,
                            'trade_count': 0
                        }
                        
                        await websocket.send(json.dumps({
                            "method": "subscribeTokenTrade",
                            "keys": [token_mint]
                        }))
                        
                    elif "trade" in data:
                        trade = data["trade"]
                        mcap = 0
                        if "market_cap" in trade:
                            mcap = float(trade["market_cap"])
                        elif "marketCap" in trade:
                            mcap = float(trade["marketCap"])
                        elif "marketCapSol" in trade:
                            mcap = float(trade["marketCapSol"]) * self.SOL_PRICE
                        
                        if mcap > 0:  # Only process if we got a valid mcap
                            await self.process_trade(trade["mint"], mcap)
                        
                except Exception as e:
                    self.logger.log(f"Error processing message: {str(e)}")
                    self.logger.log(f"Message data: {data}")
    
    async def process_trade(self, token_mint, mcap):
        try:
            if token_mint not in self.active_tokens or token_mint in self.saved_tokens:
                return
                
            token_data = self.active_tokens[token_mint]
            current_time = time.time()
            elapsed = current_time - token_data['start_time']
            
            token_data['trade_count'] += 1
            
            if mcap > token_data['highest_mcap']:
                token_data['highest_mcap'] = mcap
                self.logger.log(f"New high for {token_mint}: ${mcap:,.2f} at {elapsed:.2f}s")
            
            # Check for pump within 20 seconds
            if elapsed <= 20:
                if mcap >= 15000:
                    pump_msg = f"\nPUMP DETECTED!\nToken: {token_mint}\nInitial MCap: ${token_data['initial_mcap']:,.2f}\nPump MCap: ${mcap:,.2f}\nTime: {elapsed:.2f}s"
                    self.logger.log(pump_msg)
                    
                    await self.save_pump(token_mint, mcap, elapsed, token_data['initial_mcap'])
                    self.saved_tokens.add(token_mint)
                    del self.active_tokens[token_mint]
            
            # Clean up old tokens
            elif elapsed > 20:
                if token_mint not in self.saved_tokens:
                    self.logger.log(f"Token {token_mint} expired after {token_data['trade_count']} trades. Highest mcap: ${token_data['highest_mcap']:,.2f}")
                del self.active_tokens[token_mint]
                
        except Exception as e:
            self.logger.log(f"Error in process_trade: {str(e)}")
    
    async def save_pump(self, token_mint, mcap, elapsed, initial_mcap):
        filename = f'pumps_{datetime.now().strftime("%Y%m%d")}.json'
        
        pump_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'token': token_mint,
            'initial_mcap': initial_mcap,
            'pump_mcap': mcap,
            'seconds_to_pump': elapsed
        }
        
        try:
            with open(filename, 'a') as f:
                json.dump(pump_data, f)
                f.write('\n')
            self.logger.log(f"Successfully archived pump for {token_mint}")
        except Exception as e:
            self.logger.log(f"Error saving pump data: {str(e)}")

async def main():
    detector = PumpPortalDetector()
    try:
        await detector.monitor_new_tokens()
    except Exception as e:
        detector.logger.log(f"Main loop error: {str(e)}")
        import traceback
        detector.logger.log(traceback.format_exc())

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    asyncio.run(main())