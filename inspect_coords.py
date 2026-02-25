import pickle as pkl
import numpy as np

# ============ CHANGE THIS PATH TO YOUR PKL FILE ============
PKL_PATH = "/home/wsi3567/Point-Policy/Franka-Teach/data/processed_data_pkl/expert_demos/franka_env/pingpong2.pkl"

# ============ WHICH DEMOS TO INSPECT ============
DEMO_INDICES = [10, 11, 12, 13]

# ============ LOAD DATA ============
with open(PKL_PATH, "rb") as f:
    data = pkl.load(f)

observations = data["observations"]
print(f"Total demos in pkl: {len(observations)}")
print(f"Keys in first demo: {list(observations[0].keys())}")
print("=" * 80)

# ============ FIGURE OUT WHICH KEYS EXIST ============
sample = observations[0]
# Find all track keys
track_keys = [k for k in sample.keys() if "tracks" in k]
print(f"Available track keys: {track_keys}")
print("=" * 80)

for demo_idx in DEMO_INDICES:
    if demo_idx >= len(observations):
        print(f"\n*** Demo {demo_idx} does not exist (only {len(observations)} demos) ***")
        continue

    obs = observations[demo_idx]
    print(f"\n{'#' * 80}")
    print(f"DEMO {demo_idx}")
    print(f"{'#' * 80}")

    for pixel_key_suffix in ["pixels1", "pixels2"]:
        # Try different key naming conventions
        robot_3d_key = f"robot_tracks_3d_{pixel_key_suffix}"
        object_3d_key = f"object_tracks_3d_{pixel_key_suffix}"
        human_3d_key = f"human_tracks_3d_{pixel_key_suffix}"
        robot_2d_key = f"robot_tracks_{pixel_key_suffix}"
        object_2d_key = f"object_tracks_{pixel_key_suffix}"
        human_2d_key = f"human_tracks_{pixel_key_suffix}"

        print(f"\n--- {pixel_key_suffix} ---")

        ################################################################
        # HAND / ROBOT 3D TRACKS
        ################################################################
        for key_name, label in [
            (robot_3d_key, "ROBOT (hand) 3D"),
            (human_3d_key, "HUMAN (hand) 3D"),
        ]:
            if key_name in obs:
                tracks = obs[key_name]  # shape: (T, N, 3)
                print(f"\n  {label} [{key_name}]")
                print(f"    Shape: {tracks.shape}  (timesteps, num_points, dims)")

                # Find the timestep where Z is at its LOWEST (most negative or smallest)
                # Z values per timestep: take the mean Z across all points
                mean_z_per_step = tracks[:, :, 2].mean(axis=1)
                min_z_step = np.argmin(mean_z_per_step)

                print(f"    Z range across all steps: min={tracks[:,:,2].min():.4f}, max={tracks[:,:,2].max():.4f}")
                print(f"    *** Lowest mean Z at timestep {min_z_step} (mean Z = {mean_z_per_step[min_z_step]:.4f}) ***")
                print(f"    Coordinates at timestep {min_z_step}:")
                for pt_idx in range(tracks.shape[1]):
                    x, y, z = tracks[min_z_step, pt_idx]
                    print(f"      Point {pt_idx}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

                # Also print per-point lowest Z
                print(f"    Per-point lowest Z:")
                for pt_idx in range(tracks.shape[1]):
                    z_vals = tracks[:, pt_idx, 2]
                    min_step = np.argmin(z_vals)
                    print(f"      Point {pt_idx}: lowest Z={z_vals[min_step]:.4f} at step {min_step}, coords=({tracks[min_step, pt_idx, 0]:.4f}, {tracks[min_step, pt_idx, 1]:.4f}, {tracks[min_step, pt_idx, 2]:.4f})")

        ################################################################
        # OBJECT 3D TRACKS
        ################################################################
        if object_3d_key in obs:
            tracks = obs[object_3d_key]
            print(f"\n  OBJECT 3D [{object_3d_key}]")
            print(f"    Shape: {tracks.shape}")
            print(f"    Z range: min={tracks[:,:,2].min():.4f}, max={tracks[:,:,2].max():.4f}")

            mean_z_per_step = tracks[:, :, 2].mean(axis=1)
            min_z_step = np.argmin(mean_z_per_step)
            print(f"    *** Lowest mean Z at timestep {min_z_step} (mean Z = {mean_z_per_step[min_z_step]:.4f}) ***")
            print(f"    Coordinates at timestep {min_z_step}:")
            for pt_idx in range(tracks.shape[1]):
                x, y, z = tracks[min_z_step, pt_idx]
                print(f"      Point {pt_idx}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

        ################################################################
        # 2D TRACKS (for reference)
        ################################################################
        for key_name, label in [
            (robot_2d_key, "ROBOT (hand) 2D"),
            (human_2d_key, "HUMAN (hand) 2D"),
            (object_2d_key, "OBJECT 2D"),
        ]:
            if key_name in obs:
                tracks = obs[key_name]
                print(f"\n  {label} [{key_name}]")
                print(f"    Shape: {tracks.shape}")
