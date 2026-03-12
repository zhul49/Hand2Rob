import pickle
import cv2
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
pkl_path = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/crisp.pkl"
output_dir = "/home/wsi3567/Point-Policy/videos_pkl/crisp"
cameras_to_save = ['pixels1', 'pixels2']  # Save both camera views
fps = 10

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"Loading {pkl_path}...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

num_observations = len(data['observations'])
print(f"Found {num_observations} observations")

# Loop through all observations
for obs_idx in range(num_observations):
    print(f"\n{'='*60}")
    print(f"Processing observation {obs_idx}/{num_observations}")
    print(f"{'='*60}")
    
    episode = data['observations'][obs_idx]
    
    # Process both camera views
    for camera_to_view in cameras_to_save:
        print(f"  Camera: {camera_to_view}")
        
        # Setup keys
        if camera_to_view == 'pixels1':
            img_key = 'pixels1'
            obj_key = 'object_tracks_pixels1'
            rob_key = 'robot_tracks_pixels1'
        else:
            img_key = 'pixels2'
            obj_key = 'object_tracks_pixels2'
            rob_key = 'robot_tracks_pixels2'
        
        # Check if video exists
        if img_key not in episode:
            print(f"  [WARNING] No video found for {img_key}, skipping...")
            continue
        
        video_data = episode[img_key]
        num_frames = video_data.shape[0]
        height = video_data.shape[1]
        width = video_data.shape[2]
        
        print(f"  Frames: {num_frames}, Size: {width}x{height}")
        
        # Load points
        obj_points = episode.get(obj_key, None)
        rob_points = episode.get(rob_key, None)
        
        # Output filename
        output_filename = f"demo_{obs_idx:03d}_{camera_to_view}.mp4"
        save_path = os.path.join(output_dir, output_filename)
        
        # Video writer
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Render frames
        for i in range(num_frames):
            # Get frame
            frame = video_data[i]
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw green dots (object/bottle)
            if obj_points is not None:
                pts = obj_points[i]
                for pt in pts:
                    try:
                        cv2.circle(bgr_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                    except:
                        pass
            
            # Draw red dots (robot/hand)
            if rob_points is not None:
                pts = rob_points[i]
                for pt in pts:
                    try:
                        cv2.circle(bgr_frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                    except:
                        pass
            
            out.write(bgr_frame)
        
        out.release()
        print(f"  Saved: {output_filename}")

print(f"\n{'='*60}")
print(f"ALL DONE! Saved {num_observations * len(cameras_to_save)} videos to:")
print(f"  {output_dir}")
print(f"{'='*60}")
# Run it
