import json
from datetime import datetime

def trim_data(target_mcap=38000):
    # Load price history
    with open('price_history_1.json', 'r') as f:
        price_data = json.load(f)
    
    # Load enhanced data
    with open('enhanced_1.json', 'r') as f:
        enhanced_data = json.load(f)
    
    # Find cutoff point from bottom (older transactions) up
    cutoff_index = None
    cutoff_timestamp = None
    for i in range(len(price_data)-1, -1, -1):  # Start from bottom
        if price_data[i]['market_cap'] >= target_mcap and price_data[i]['sol_amount'] > 0.1:
            cutoff_index = i
            cutoff_timestamp = price_data[i]['timestamp']
            break
    
    if cutoff_index is None:
        print("No transaction found exceeding target market cap")
        return
        
    # Keep only data before (and including) the cutoff point
    trimmed_price = price_data[cutoff_index:]
    
     # Find corresponding enhanced data cutoff
    trimmed_enhanced = [window for window in enhanced_data 
                       if window['timestamp'] < cutoff_timestamp]
    
    # Save trimmed data
    with open('price_history_8n6qNhEmzaikdk398Dd2GZ9XksPNwfPgBeYCb5b4XAjy_20241028_232515.json', 'w') as f:
        json.dump(trimmed_price, f, indent=2)
        
    with open('enhanced_8n6qNhEmzaikdk398Dd2GZ9XksPNwfPgBeYCb5b4XAjy_20241028_232515.json', 'w') as f:
        json.dump(trimmed_enhanced, f, indent=2)
    
    print(f"Found cutoff at timestamp: {datetime.fromtimestamp(cutoff_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Original price history entries: {len(price_data)}")
    print(f"Trimmed price history entries: {len(trimmed_price)}")
    print(f"Original enhanced windows: {len(enhanced_data)}")
    print(f"Trimmed enhanced windows: {len(trimmed_enhanced)}")
    print(f"\nFirst market cap in trimmed data: ${trimmed_price[0]['market_cap']:,.2f}")
    print(f"Last market cap in trimmed data: ${trimmed_price[-1]['market_cap']:,.2f}")

if __name__ == "__main__":
    trim_data()