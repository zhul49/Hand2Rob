import pickle
import numpy as np
import shutil
import os

# --- CONFIGURATION ---
# Path to your data
pkl_path = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/cup_stack.pkl"

# Resolution Settings
ORIGINAL_W, ORIGINAL_H = 640.0, 480.0  # RealSense Raw
TARGET_W, TARGET_H     = 256.0, 256.0  # Point-Policy Training Size

# Calculate Scale Factors
scale_x = TARGET_W / ORIGINAL_W
scale_y = TARGET_H / ORIGINAL_H

print(f"--- SCALING FIXER ---")
print(f"Target: {pkl_path}")
print(f"Scaling Factor: X={scale_x:.3f}, Y={scale_y:.3f}")

# 1. Backup
backup_path = pkl_path + ".backup_full"
if not os.path.exists(backup_path):
    print(f"Creating backup at {backup_path}...")
    shutil.copyfile(pkl_path, backup_path)
else:
    print(f"Backup already exists at {backup_path}")

# 2. Load Data
print("Loading data...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

episodes = data['observations']
fixes = 0

# 3. Apply Fix
# We look for ANY key that contains point data
keys_to_check = [
    'robot_tracks_pixels1', 'robot_tracks_pixels2',   # Red Dots (Hand)
    'object_tracks_pixels1', 'object_tracks_pixels2'  # Green Dots (Cup)
]

for i, ep in enumerate(episodes):
    for key in keys_to_check:
        if key in ep:
            pts = ep[key]

            # --- SAFETY CHECK ---
            # Only scale if we see coordinates larger than the target image (256)
            # This prevents double-scaling if you run the script twice.
            if np.max(pts) > 256:
                # Apply Scaling
                pts[:, :, 0] *= scale_x
                pts[:, :, 1] *= scale_y

                # Save back to episode
                ep[key] = pts
                fixes += 1
                # print(f"  Fixed {key} in Episode {i}")

print(f"\nSUCCESS: Applied scaling to {fixes} data arrays.")

# 4. Save
print("Saving file...")
with open(pkl_path, "wb") as f:
    pickle.dump(data, f)

print("Done! Both Red and Green dots fit inside the video now.")
