# © 2024 Aristotle University of Thessaloniki, Greece Chariton Keramarakis
# All Rights Reserved.
# 
# This code is proprietary and confidential. It is licensed only for non-commercial, academic use 
# at Aristotle University of Thessaloniki, or with express written permission from the author. 
# Unauthorized copying, modification, or distribution of this code is strictly prohibited.
# 
# Licensed under the Custom License Agreement for Non-Commercial Academic Use.
# See the LICENSE file for details.
import numpy as np
import json
from typing import List, Dict, Union
from collections import defaultdict

def classify_token_pattern(price_history: List[Dict]) -> Dict[str, Union[bool, float]]:
    """
    Classifies if a token's price history matches the pattern of steady rise followed by sharp dump.
    Enhanced to detect artificial vs organic pump patterns.
    """
    if not price_history or len(price_history) < 10:
        return {"is_pump_dump": False, "reason": "Insufficient data points"}

    # Extract price and market cap data
    prices = np.array([float(p['price_in_sol']) for p in price_history])
    market_caps = np.array([float(p['market_cap']) for p in price_history])
    timestamps = np.array([float(p['timestamp']) for p in price_history])
    
    # Find peaks
    price_peak_idx = np.argmax(prices)
    mcap_peak_idx = np.argmax(market_caps)
    
    # Use the later peak for analysis
    peak_idx = max(price_peak_idx, mcap_peak_idx)

    # Calculate metrics
    metrics = calculate_metrics(price_history, prices, market_caps, timestamps, peak_idx)
    
    # Add timestamp information
    metrics.update({
        'peak_time': price_history[peak_idx]['date'],
        'start_time': price_history[0]['date'],
        'end_time': price_history[-1]['date'],
        'peak_price_sol': float(prices[peak_idx]),
        'peak_market_cap': float(market_caps[peak_idx]),
        'final_price_sol': float(prices[-1]),
        'final_market_cap': float(market_caps[-1]),
        'peak_tx_signature': price_history[peak_idx]['signature']
    })

    # Classification
    is_pump_dump = True
    reason = "Matches pump and dump pattern"

    # Enhanced classification criteria
    if metrics['growth_duration_minutes'] < 1:
        is_pump_dump = False
        reason = "Price increase too rapid to be organic"
    elif metrics['tx_spacing_score'] < 0.7:
        is_pump_dump = False
        reason = "Transaction pattern indicates artificial trading"
    elif metrics['price_stability'] < 0.6:
        is_pump_dump = False
        reason = "Price movement pattern too volatile/vertical"
    elif metrics['mcap_increase_factor'] < 5:
        is_pump_dump = False
        reason = "Market cap increase not significant enough"
    elif metrics['dump_magnitude'] < 0.7:
        is_pump_dump = False
        reason = "Dump not significant enough"

    return {
        "is_pump_dump": is_pump_dump,
        "reason": reason,
        "metrics": metrics
    }

def calculate_metrics(price_history: List[Dict], prices: np.ndarray, market_caps: np.ndarray, 
                     timestamps: np.ndarray, peak_idx: int) -> Dict[str, float]:
    """Calculate comprehensive metrics for pattern analysis."""
    
    def analyze_transaction_spacing(price_history, window_size=5):
        """
        Analyze transaction distribution and spacing to detect artificial trading patterns
        Returns a score between 0 and 1, where higher scores indicate more natural trading patterns
        """
        tx_windows = defaultdict(list)
        
        for entry in price_history:
            window_key = int(entry['timestamp'] // window_size) * window_size
            tx_windows[window_key].append(entry)
        
        # Calculate average time between transactions per window
        avg_spacing = []
        tx_counts = []
        for window, transactions in tx_windows.items():
            tx_counts.append(len(transactions))
            if len(transactions) > 1:
                times = sorted([tx['timestamp'] for tx in transactions])
                spacings = np.diff(times)
                avg_spacing.append(np.mean(spacings))
        
        spacing_score = 1.0
        
        # Penalize very high transaction counts in windows
        max_tx = max(tx_counts) if tx_counts else 0
        if max_tx > 20:  # More than 20 transactions in 5 seconds
            spacing_score *= 0.5
        
        # Penalize very small average spacing
        if avg_spacing:
            mean_spacing = np.mean(avg_spacing)
            if mean_spacing < 0.5:  # Less than 0.5 seconds between transactions
                spacing_score *= 0.5
            
        # Penalize highly variable transaction counts
        if tx_counts:
            tx_std = np.std(tx_counts)
            if tx_std > 10:  # High variation in transaction counts
                spacing_score *= 0.7
        
        return spacing_score

    def calculate_price_stability(prices, timestamps, window_size=30):
        """
        Analyze price movement pattern stability
        Looks for steady rises with healthy consolidation periods
        """
        if len(prices) < window_size:
            return 0.0
        
        # Calculate rolling window metrics
        stability_scores = []
        for i in range(len(prices) - window_size):
            window_prices = prices[i:i+window_size]
            window_times = timestamps[i:i+window_size]
            
            # Calculate price changes
            price_changes = np.diff(window_prices) / window_prices[:-1]
            time_changes = np.diff(window_times)
            
            # Calculate velocities
            velocities = price_changes / time_changes
            
            # Score components:
            # 1. Consistent upward movement
            up_moves = np.sum(price_changes > 0)
            
            # 2. Healthy consolidation periods
            small_moves = np.sum(np.abs(price_changes) < 0.1)  # Small price changes
            
            # 3. Penalize extreme moves
            extreme_moves = np.sum(np.abs(price_changes) > 0.3)
            
            # 4. Look for stair-step pattern
            stairs = 0
            for j in range(len(price_changes)-1):
                if price_changes[j] > 0 and abs(price_changes[j+1]) < 0.1:
                    stairs += 1
            
            # Combine scores
            window_score = (
                (up_moves + small_moves + stairs) / 
                (3 * len(price_changes))
            )
            
            # Penalize extreme movements
            window_score *= (1 - extreme_moves/len(price_changes))
            
            stability_scores.append(window_score)
            
        return np.mean(stability_scores) if stability_scores else 0.0

    def calculate_volume_concentration(price_history, timestamps, peak_idx):
        """Analyze trading volume distribution"""
        volumes = []
        for entry in price_history:
            volumes.append(float(entry['sol_amount']))
        
        volumes = np.array(volumes)
        
        # Calculate volume concentration around peak
        peak_window = 60  # 1 minute window
        peak_volume = np.sum(volumes[max(0, peak_idx-peak_window):min(len(volumes), peak_idx+peak_window)])
        total_volume = np.sum(volumes)
        
        return peak_volume / total_volume if total_volume > 0 else 0

    # Calculate time-based metrics
    total_duration = timestamps[-1] - timestamps[0]
    time_to_peak = timestamps[peak_idx] - timestamps[0]
    growth_duration_minutes = time_to_peak / 60
    
    # Basic metrics
    price_increase_factor = prices[peak_idx] / prices[0] if prices[0] != 0 else 0
    mcap_increase_factor = market_caps[peak_idx] / market_caps[0] if market_caps[0] != 0 else 0
    dump_magnitude = (prices[peak_idx] - np.min(prices[peak_idx:])) / prices[peak_idx] if prices[peak_idx] != 0 else 0
    
    # Advanced pattern analysis
    tx_spacing_score = analyze_transaction_spacing(price_history)
    price_stability = calculate_price_stability(prices, timestamps)
    volume_concentration = calculate_volume_concentration(price_history, timestamps, peak_idx)
    
    # Time-based adjustments
    if total_duration < 300:  # Less than 5 minutes total
        tx_spacing_score *= 0.5
        price_stability *= 0.5
    
    if time_to_peak < 180:  # Peak reached in less than 3 minutes
        price_stability *= 0.3
    
    return {
        "price_increase_factor": float(price_increase_factor),
        "mcap_increase_factor": float(mcap_increase_factor),
        "growth_duration_minutes": float(growth_duration_minutes),
        "tx_spacing_score": float(tx_spacing_score),
        "price_stability": float(price_stability),
        "dump_magnitude": float(dump_magnitude),
        "volume_concentration": float(volume_concentration),
        "time_to_peak_ratio": float(peak_idx / len(prices)),
        "peak_market_cap_usd": float(market_caps[peak_idx]),
        "final_market_cap_usd": float(market_caps[-1]),
        "market_cap_change_percent": float((market_caps[-1] - market_caps[0]) / market_caps[0] * 100) if market_caps[0] != 0 else 0,
        "transactions_per_minute": float(len(price_history) / (total_duration/60))
    }

def main():
    # Load the price history
    with open('price_history_simple.json', 'r') as f:
        price_history = json.load(f)
    
    # Sort by timestamp
    price_history.sort(key=lambda x: x['timestamp'])
    
    # Analyze the pattern
    result = classify_token_pattern(price_history)
    
    # Print results
    print("\nToken Pattern Analysis Results:")
    print("-" * 40)
    print(f"Is Pump and Dump: {result['is_pump_dump']}")
    print(f"Reason: {result['reason']}")
    print("\nDetailed Metrics:")
    print("-" * 40)
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            if 'market_cap' in key.lower():
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # Save analysis results
    with open('token_analysis_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to token_analysis_results.json")

if __name__ == "__main__":
    main()