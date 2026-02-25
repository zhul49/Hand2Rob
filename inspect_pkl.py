import pickle
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
pkl_path = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/pingpong.pkl"
camera_to_view = 'pixels1'  # Change to 'pixels1' or 'pixels2' to switch views

print(f"Loading {pkl_path}...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Get the first episode
episode = data['observations'][1]

# --- SETUP KEYS ---
# Automatically matching keys based on which camera you selected
if camera_to_view == 'pixels1':
    img_key = 'pixels1'
    obj_key = 'object_tracks_pixels1'   # Green Dots (Cup)
    rob_key = 'robot_tracks_pixels1'    # Red Dots (Hand)
else:
    img_key = 'pixels2'
    obj_key = 'object_tracks_pixels2'
    rob_key = 'robot_tracks_pixels2'

# Verify Data Exists
if img_key not in episode:
    print(f"ERROR: Could not find video key '{img_key}'")
    print(f"Available keys: {list(episode.keys())}")
    exit()

video_data = episode[img_key]
num_frames = video_data.shape[0]
height = video_data.shape[1]
width = video_data.shape[2]

print(f"\n[INFO] Viewing {camera_to_view}")
print(f"[INFO] Video Shape: {video_data.shape}")

# Load Points (Handle missing keys gracefully)
obj_points = episode.get(obj_key, None)
rob_points = episode.get(rob_key, None)

if obj_points is None: print(f"[WARNING] No Green Dots found for {obj_key}")
if rob_points is None: print(f"[WARNING] No Red Dots found for {rob_key}")

# --- VIDEO WRITER ---
save_path = f"debug_robot_{camera_to_view}.mp4"
fps = 10
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Rendering {num_frames} frames to '{save_path}'...")

for i in range(num_frames):
    # 1. Get Frame
    frame = video_data[i]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 2. Draw Green Dots (Object)
    if obj_points is not None:
        pts = obj_points[i]
        for pt in pts:
            try:
                # Draw Green Circle
                cv2.circle(bgr_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            except: pass

    # 3. Draw Red Dots (Robot/Hand)
    if rob_points is not None:
        pts = rob_points[i]
        for pt in pts:
            try:
                # Draw Red Circle
                cv2.circle(bgr_frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
            except: pass

    out.write(bgr_frame)

out.release()
print(f"Done! Video saved to {save_path}")
