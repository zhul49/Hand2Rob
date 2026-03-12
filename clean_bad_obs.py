#!/usr/bin/env python3
"""
Remove bad observations from bottle_flip_real_1.pkl
Removes indices: 0, 1, 7 (observations with REFLECT errors)
Can be run from anywhere.
"""

import pickle as pkl
import sys
from pathlib import Path
import os

# Configuration
REMOVE_INDICES = [1, 7, 10, 24, 26, 29, 30, 33, 35, 37, 38, 39, 40, 42, 43, 44, 47, 48, 50, 52, 62]

# Try to find the file in multiple locations
POSSIBLE_PATHS = [
    Path.cwd() / 'egg.pkl',  # Current directory
    Path.home() / 'Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/egg.pkl',
    Path('/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/egg.pkl'),
    Path('/home/robert/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/egg.pkl'),
]

def find_file():
    """Find the pkl file in known locations"""
    for path in POSSIBLE_PATHS:
        if path.exists():
            return path
    return None

def main():
    # Find the input file
    input_path = find_file()
    
    if input_path is None:
        print("❌ Error: Could not find egg.pkl")
        print("\nSearched in:")
        for path in POSSIBLE_PATHS:
            print(f"  - {path}")
        print("\nYou can also provide the path as an argument:")
        print(f"  python {sys.argv[0]} /path/to/egg.pkl")
        sys.exit(1)
    
    # Allow override via command line
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if not input_path.exists():
            print(f"❌ Error: {input_path} not found!")
            sys.exit(1)
    
    # Set output path in same directory as input
    output_path = input_path.parent / 'egg_cleaned.pkl'
    backup_path = input_path.parent / 'egg_backup.pkl'
    
    print(f"Found file: {input_path}")
    
    # Load the original data
    print(f"Loading {input_path.name}...")
    with open(input_path, 'rb') as f:
        data = pkl.load(f)
    
    print(f"Original number of observations: {len(data['observations'])}")
    
    # Remove bad observations
    print(f"Removing observations: {REMOVE_INDICES}")
    
    new_observations = []
    for i, obs in enumerate(data['observations']):
        if i not in REMOVE_INDICES:
            new_observations.append(obs)
        else:
            print(f"  Removed observation {i}")
    
    print(f"New number of observations: {len(new_observations)}")
    
    # Update the data dictionary
    data['observations'] = new_observations
    
    # Save to new file
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pkl.dump(data, f)
    
    print("✅ Done!")
    print(f"\nTo use the cleaned version, run:")
    print(f"  mv {input_path} {backup_path}")
    print(f"  mv {output_path} {input_path}")
    print(f"\nOr from anywhere:")
    print(f"  cd {input_path.parent}")
    print(f"  mv egg.pkl egg_backup.pkl")
    print(f"  mv egg_cleaned.pkl egg.pkl")

if __name__ == "__main__":
    main()
