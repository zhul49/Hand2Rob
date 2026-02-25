import pickle
import numpy as np

# Load data
pkl_path = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/cup_stack.pkl"
print(f"Loading {pkl_path}...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

episode = data['observations'][0]

# Check Camera 2 Red Dots (Robot)
key = 'robot_tracks_pixels2'

if key not in episode:
    print(f"ERROR: Key '{key}' not found!")
    print("Available keys:", list(episode.keys()))
else:
    points = episode[key] # Shape (Time, 8, 2)
    
    print("\n--- COORDINATE CHECK ---")
    print(f"Image Size in file: {episode['pixels2'].shape}")
    
    # Check first 5 frames
    for i in range(5):
        pts = points[i] # 8 dots
        print(f"Frame {i}:")
        for j, pt in enumerate(pts):
            x, y = pt[0], pt[1]
            status = "OK"
            if x < 0 or x > 256 or y < 0 or y > 256:
                status = "OFF-SCREEN"
            print(f"  Dot {j+1}: ({x:.1f}, {y:.1f}) -> {status}")
