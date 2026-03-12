import pickle
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# Path to your processed data file
pkl_path = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/cup_tactile.pkl"
output_video_name = "cup_tactile_hand.mp4"
# ---------------------

def visualize_pkl(path):
    if not os.path.exists(path):
        print(f"❌ Error: File not found at {path}")
        return

    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Get the first demo
    obs = data['observations'][49]
    
    # Check if we have the data we need
    if 'pixels2' not in obs or 'human_tracks_pixels2' not in obs:
        print("❌ Error: Could not find 'pixels1' or 'human_tracks_pixels1' in the data.")
        return

    images = obs['pixels1']        # The video frames (256x256)
    points = obs['human_tracks_pixels1']  # The tracked points (Time, Points, 2)
    
    # Setup Video Writer
    h, w, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, 30.0, (w, h))

    print(f"Rendering video with {len(images)} frames...")

    for i in range(len(images)):
        # 1. Get image (Convert RGB to BGR for OpenCV)
        img = images[i].copy()
        frame_points = points[i]
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 2. Get points for this frame
        #frame_points = points[i] # Shape: (Num_Points, 2)

        # 3. Draw the points
        for pt in frame_points:
            x, y = int(pt[0]), int(pt[1])
            # Draw a bright GREEN circle
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1) 
            # Draw a black outline so you can see it on white cups
            cv2.circle(img, (x, y), 4, (0, 0, 0), 1)

        # 4. Write to video
        out.write(img)

    out.release()
    print(f"✅ Done! Video saved to: {os.path.abspath(output_video_name)}")
    print("You can download this file to your laptop to watch it.")

if __name__ == "__main__":
    visualize_pkl(pkl_path)
