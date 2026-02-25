import cv2
import numpy as np
import pickle as pkl
from pathlib import Path

# --- CONFIG ---
# Update this to match the path where your script saved the file
pkl_path = "~/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/cup_stack.pkl" 
camera_name = "cam_1" # Or cam_2
# ----------------

def visualize_data(pkl_file, cam_name):
    data = pkl.load(open(pkl_file, "rb"))
    
    # Map camera name to the key used in your dictionary
    # (Assuming standard setup: cam_1 -> image, cam_2 -> image_2, etc. 
    # Adjust based on your 'camera2pixelkey' from utils)
    if cam_name == "cam_1":
        img_key = "image"
        track_key = "robot_tracks_image"
    else:
        img_key = "image_right" # or whatever your key mapping is
        track_key = f"robot_tracks_{img_key}"

    print(f"Visualizing {len(data['observations'])} frames...")
    print("Press 'q' to quit, any other key for next frame.")

    for i, obs in enumerate(data["observations"]):
        # 1. Get the image (it is already resized to 256x256 in your script)
        # Note: Depending on how it was saved, it might be [C, H, W] or [H, W, C]
        # Your script implies a list of images, so likely [H, W, C] is maintained if using cv2
        img = obs[img_key][0] # Taking the first frame in the chunk if stacked, or just the image
        
        # Ensure image is uint8 for cv2
        img = np.array(img, dtype=np.uint8)
        
        # Make a copy to draw on
        canvas = img.copy()

        # 2. Get the projected 2D robot points
        # shape: (N_points, 2)
        points_2d = obs[track_key][0] 

        # 3. Draw the points
        # Point 0 is usually the base/TCP, draw it distinct (Red)
        if len(points_2d) > 0:
            cx, cy = int(points_2d[0][0]), int(points_2d[0][1])
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1) # Red center

        # Draw the rest of the gripper points (Green)
        for j in range(1, len(points_2d)):
            px, py = int(points_2d[j][0]), int(points_2d[j][1])
            cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)

        # 4. Show
        cv2.imshow(f"Check: {cam_name}", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        key = cv2.waitKey(0) # Wait for keypress
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_data(pkl_path, camera_name)
