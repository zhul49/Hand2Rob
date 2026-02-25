import pickle
import numpy as np

# Configuration
task_name = "cup_stack"
pickle_path = f"/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/{task_name}.pkl"
traj_idx = 0  # Change this if you are working on a different demo

# 1. Load the file
try:
    with open(pickle_path, 'rb') as f:
        saved_data = pickle.load(f)
    print(f"✅ Successfully loaded {pickle_path}")
except FileNotFoundError:
    print(f"❌ File not found: {pickle_path}")
    saved_data = None

if saved_data:
    # 2. Get the specific trajectory
    if traj_idx < len(saved_data['observations']):
        obs = saved_data['observations'][traj_idx]
        print(f"\n--- Inspection for Trajectory {traj_idx} ---")
        
        # 3. Check for specific keys
        # These are the most likely names for your points
        keys_to_check = [
            'pixels1', 
            'pixels2', 
            'pixels1_points', 
            'pixels2_points',
            'human_tracks_3d_pixels1',
            'human_tracks_3d_pixels2'
        ]
        
        found_any = False
        for key in keys_to_check:
            if key in obs:
                val = obs[key]
                # Format output based on type
                if isinstance(val, (np.ndarray, list)):
                    print(f"🟢 [FOUND] {key}: Shape {np.shape(val)}")
                    # print(val) # Uncomment to see exact numbers
                else:
                    print(f"🟢 [FOUND] {key}: {val}")
                found_any = True
            else:
                print(f"🔴 [MISSING] {key}")
        
        if not found_any:
            print("\n⚠️ No point data found for this trajectory. It is clean and ready to be re-annotated.")
        else:
            print("\n⚠️ Points exist! If you want to redo them, run the DELETE code I gave you above.")
            
    else:
        print(f"❌ traj_idx {traj_idx} is out of bounds. File only has {len(saved_data['observations'])} trajectories.")
