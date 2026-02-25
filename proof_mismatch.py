import numpy as np
import pickle

# Paths
calib_path = '/home/wsi3567/Point-Policy/calib.npy'
data_path = '/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/cup_stack.pkl'

print("--- PROOF OF MISMATCH ---")

# 1. Check Calibration Keys
calib = np.load(calib_path, allow_pickle=True).item()
print(f"Calibration Keys: {list(calib.keys())}")

# 2. Check Data Keys
with open(data_path, "rb") as f:
    data = pickle.load(f)
    episode_keys = list(data['observations'][0].keys())
    
    # Filter for just pixel keys to make it readable
    pixel_keys = [k for k in episode_keys if 'pixels' in k]
    print(f"Data Keys:        {pixel_keys}")

print("\n--- CONCLUSION ---")
if 'pixels2' in pixel_keys and 'pixels2' not in calib:
    print("MISMATCH FOUND: Data has 'pixels2', but Calibration DOES NOT.")
    print("This is why the robot ignores Camera 2.")
else:
    print("Keys match.")
