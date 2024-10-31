import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Constants
SOL_TO_USD = 170

def plot_token_price_history(json_data):
    # Convert JSON string to dict if needed
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Create DataFrame
    df = pd.DataFrame(data['price_history'])
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Calculate USD price
    df['price_usd'] = df['price_in_sol'] * SOL_TO_USD
    
    # Create the figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    fig.suptitle('Token Price History and Volume', fontsize=16)
    
    # Plot price
    ax1.plot(df['datetime'], df['price_usd'], 'b-', linewidth=2)
    ax1.set_ylabel('Price (USD)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price over Time')
    
    # Format y-axis to scientific notation
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot volume
    ax2.bar(df['datetime'], df['token_amount'], color='gray', alpha=0.5)
    ax2.set_ylabel('Volume (Token Amount)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Trading Volume')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add some statistics as text
    stats_text = (
        f"First Trade: {data['first_trade_time']}\n"
        f"Last Trade: {data['last_trade_time']}\n"
        f"Total Trades: {data['total_points']}\n"
        f"Max Price: ${df['price_usd'].max():.8f}\n"
        f"Min Price: ${df['price_usd'].min():.8f}\n"
        f"Price Change: {((df['price_usd'].iloc[-1] - df['price_usd'].iloc[0]) / df['price_usd'].iloc[0] * 100):.2f}%"
    )
    
    # Add text box with statistics
    plt.figtext(1.02, 0.5, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Assuming your JSON data is in a file called 'price_data.json'
    with open('token_price_history.json', 'r') as f:
        data = json.load(f)
    
    fig = plot_token_price_history(data)
    plt.show()