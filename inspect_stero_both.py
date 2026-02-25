import pickle
import cv2
import numpy as np

# --- CONFIGURATION ---
pkl_path = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/cup_stack.pkl"
save_path = "debug_stereo_view.mp4"

print(f"Loading {pkl_path}...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

episode = data['observations'][0]

# --- CHECK FOR BOTH CAMERAS ---
if 'pixels1' not in episode or 'pixels2' not in episode:
    print("ERROR: This data does not have both 'pixels1' and 'pixels2'.")
    exit()

# Get Video Data
vid1 = episode['pixels1']
vid2 = episode['pixels2']

# Get Points (Handle missing keys gracefully)
rob1 = episode.get('robot_tracks_pixels1', None) # Red
obj1 = episode.get('object_tracks_pixels1', None) # Green
rob2 = episode.get('robot_tracks_pixels2', None) # Red
obj2 = episode.get('object_tracks_pixels2', None) # Green

# --- THE FIX IS HERE ---
# We explicitly unpack all 4 dimensions: (Frames, Height, Width, Channels)
num_frames, h, w, _ = vid1.shape

print(f"Video Shape: {vid1.shape}")
print(f"Rendering Side-by-Side Stereo View to {save_path}...")

# Setup Side-by-Side Video (Width is doubled)
fps = 10
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))

for i in range(num_frames):
    # --- PREPARE CAMERA 1 ---
    frame1 = vid1[i].astype(np.uint8)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    
    # Draw Dots Cam 1
    if rob1 is not None:
        for pt in rob1[i]: 
            try: cv2.circle(frame1, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
            except: pass
    if obj1 is not None:
        for pt in obj1[i]: 
            try: cv2.circle(frame1, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            except: pass
        
    cv2.putText(frame1, "CAM 1", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- PREPARE CAMERA 2 ---
    frame2 = vid2[i].astype(np.uint8)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

    # Draw Dots Cam 2
    if rob2 is not None:
        for pt in rob2[i]: 
            try: cv2.circle(frame2, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
            except: pass
    if obj2 is not None:
        for pt in obj2[i]: 
            try: cv2.circle(frame2, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            except: pass

    cv2.putText(frame2, "CAM 2", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- COMBINE & SAVE ---
    combined = np.hstack((frame1, frame2))
    out.write(combined)

out.release()
print("Done! Download 'debug_stereo_view.mp4' to check the 3D tracking.")
