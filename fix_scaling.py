import pickle
import numpy as np
import shutil

# Path to your bad data
pkl_path = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/bottle_rack.pkl"
backup_path = pkl_path + ".backup"

# 1. Backup the file just in case
print(f"Backing up to {backup_path}...")
shutil.copyfile(pkl_path, backup_path)

print(f"Loading {pkl_path}...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# 2. Define the Scaling Factors
# Converting FROM RealSense (640x480) TO Training Size (256x256)
ORIGINAL_W = 640.0
ORIGINAL_H = 480.0
TARGET_W = 256.0
TARGET_H = 256.0

scale_x = TARGET_W / ORIGINAL_W
scale_y = TARGET_H / ORIGINAL_H

print(f"Applying Scale X: {scale_x:.3f} | Scale Y: {scale_y:.3f}")

# 3. Patch the Points
count = 0
# The data structure is usually data['observations'] -> list of episodes
episodes = data['observations']

for i, ep in enumerate(episodes):
    # Fix Robot Points (Red Dots)
    if 'robot_tracks_pixels2' in ep:
        pts = ep['robot_tracks_pixels2'] # Shape (Time, N, 2)
        # Apply scaling
        pts[:, :, 0] *= scale_x
        pts[:, :, 1] *= scale_y
        ep['robot_tracks_pixels2'] = pts
        count += 1
        
    # Fix Camera 1 too if it exists
    if 'robot_tracks_pixels1' in ep:
        pts = ep['robot_tracks_pixels1']
        pts[:, :, 0] *= scale_x
        pts[:, :, 1] *= scale_y
        ep['robot_tracks_pixels1'] = pts

print(f"Fixed scaling for {count} episodes.")

# 4. Save the fixed file
with open(pkl_path, "wb") as f:
    pickle.dump(data, f)

print("Done! The points should now fit inside the image.")
