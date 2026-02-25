import pickle
import numpy as np

file_path = "Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/bottle_flip_real_1.pkl"

print(f"--- Deep Inspection of: {file_path} ---")

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 1. Print all keys
print(f"Keys found: {list(data.keys())}")

# 2. Locate the 'actions'
if 'actions' in data:
    actions = data['actions']
    
    # Check if it's a list (multiple demos) or one big array
    if isinstance(actions, list):
        print(f"\nFound {len(actions)} trajectories in 'actions'.")
        # Check the first trajectory
        first_demo = actions[0]
    else:
        print(f"\n'actions' is a single array of shape {actions.shape}")
        first_demo = actions

    # 3. Analyze Gripper (Last dimension)
    print(f"Shape of first demo actions: {np.shape(first_demo)}")
    
    # Usually [x, y, z, r, p, y, GRIPPER] -> Gripper is last (-1)
    gripper = first_demo[:, -1]
    
    print(f"\n--- Gripper Stats (First Demo) ---")
    print(f"First 10 values: {gripper[:10]}")
    print(f"Min: {np.min(gripper)}")
    print(f"Max: {np.max(gripper)}")
    
    if np.min(gripper) == 0 and np.max(gripper) == 0:
        print("\n🚨 CONFIRMED: Gripper values are ALL ZEROS.")
    elif np.min(gripper) == -1 and np.max(gripper) == 1:
        print("\n✅ Gripper seems normalized (-1 to 1).")
    else:
        print(f"\n⚠️ Gripper values are present but range is {np.min(gripper)} to {np.max(gripper)}.")

else:
    print("\n❌ Could not find key 'actions'. Printing type of first key:")
    first_key = list(data.keys())[0]
    print(f"Type of data['{first_key}']: {type(data[first_key])}")
