import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional

class TimeBasedPumpVisualizer:
    def __init__(self, analysis_file: str = "training_windows_analysis.json"):
        with open(analysis_file, 'r') as f:
            self.data = json.load(f)
        self.windows = pd.DataFrame(self.data['windows'])
        
    def _prepare_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamps to seconds from start"""
        # Convert timestamps to integers
        df['timestamp'] = df['timestamp'].astype(int)
        
        # Calculate relative time from first transaction
        start_time = df['timestamp'].min()
        df['seconds_from_start'] = df['timestamp'] - start_time
        
        # Sort by time
        return df.sort_values('seconds_from_start')
        
    def plot_pump_prediction(self, save_path: Optional[str] = None):
        """Plot predictions over time with pump moment marked"""
        # Prepare data
        df = self.windows.copy()
        df = self._prepare_time_series(df)
        
        # Create plot
        plt.figure(figsize=(15, 7))
        
        # Plot model predictions
        plt.plot(df['seconds_from_start'], 
                df['model_prediction'],
                label='Model Confidence',
                color='blue',
                linewidth=2)
        
        # Find pump moment (transition from 0 to 1)
        pump_moment = None
        for i in range(1, len(df)):
            if df.iloc[i-1]['assigned_label'] == 0 and df.iloc[i]['assigned_label'] == 1:
                pump_moment = df.iloc[i]['seconds_from_start']
                break
        
        # Add pump line if found
        if pump_moment:
            plt.axvline(x=pump_moment, 
                       color='red',
                       linestyle='--',
                       label=f'Pump Moment (t={pump_moment}s)',
                       alpha=0.7)
            
            # Add annotation
            plt.annotate(f'Pump Detected\nt={pump_moment}s', 
                        xy=(pump_moment, 0.5),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.7),
                        ha='left')
        
        # Add threshold line
        plt.axhline(y=0.5,
                   color='gray',
                   linestyle=':',
                   alpha=0.5,
                   label='Decision Threshold')
        
        # Styling
        plt.title('Pump Prediction Confidence Over Time', pad=20)
        plt.xlabel('Seconds from First Transaction')
        plt.ylabel('Model Prediction Confidence')
        
        # Add duration information
        total_duration = df['seconds_from_start'].max()
        plt.text(0.02, 0.98,
                f'Total Duration: {total_duration}s ({total_duration/60:.1f}m)',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Optional minutes axis on top
        ax2 = plt.gca().twiny()
        ax2.set_xlim(plt.gca().get_xlim())
        ax2.set_xticks(plt.gca().get_xticks())
        ax2.set_xticklabels([f'{x/60:.1f}' for x in plt.gca().get_xticks()])
        ax2.set_xlabel('Minutes from First Transaction')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        # Print timing analysis
        print("\nTiming Analysis:")
        print(f"Total duration: {total_duration} seconds ({total_duration/60:.1f} minutes)")
        if pump_moment:
            print(f"Time until pump: {pump_moment} seconds ({pump_moment/60:.1f} minutes)")
            print(f"Percentage of duration until pump: {(pump_moment/total_duration)*100:.1f}%")
        print(f"Number of predictions: {len(df)}")
        
        # Return timing data
        return {
            'total_duration': total_duration,
            'pump_time': pump_moment,
            'prediction_counts': len(df)
        }

def analyze_pump_timing(json_file: str = "training_windows_analysis.json"):
    """Run timing analysis"""
    visualizer = TimeBasedPumpVisualizer(json_file)
    timing_data = visualizer.plot_pump_prediction(save_path="pump_timing_analysis.png")
    return timing_data

if __name__ == "__main__":
    timing_data = analyze_pump_timing()