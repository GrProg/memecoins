import os
import json
from datetime import datetime

def sort_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Sort the data based on timestamp
    sorted_data = sorted(data, key=lambda x: x['timestamp'])
    
    # Write the sorted data back to the same file
    with open(file_path, 'w') as file:
        json.dump(sorted_data, file, indent=2)

def process_json_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            sort_json_file(file_path)
            print(f"Sorted: {filename}")

# Main execution
folder_path = 'yes'  # Replace with the actual path to your 'yes' folder

process_json_files(folder_path)

print("All JSON files have been sorted in place.")