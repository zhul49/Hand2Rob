"""
AprilTag-on-table camera extrinsic calibration for Franka.

Generates all 4 possible axis visualizations per camera.
Pick the one where the origin is at the robot base and X points forward.
Then re-run with --angle N to lock in the correct rotation.

Usage:
  python calibrate_apriltag_table.py                  # generates all options
  python calibrate_apriltag_table.py --angle 90       # locks in 90° for both
  python calibrate_apriltag_table.py --angle1 90 --angle2 270  # per-camera
"""

import cv2
from cv2 import aruco
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import argparse

# ── paths ────────────────────────────────────────────────────────────────────
SAVE_DIR = Path("/home/robert/Point-Policy/point_policy/calib")
PATH_SAVE_CALIB = SAVE_DIR / "calib.npy"

# ── cameras ──────────────────────────────────────────────────────────────────
CAM_IDS = [1, 2]

CAM_SERIALS = {
    1: "342522070195",
    2: "231522071499",
}

N_FRAMES = 30

# ── AprilTag config ──────────────────────────────────────────────────────────
TAG_FAMILY = aruco.DICT_APRILTAG_36h11
TAG_ID = 0
TAG_SIZE_M = 0.10  # 10 cm

# ── Known tag offset in robot base frame ─────────────────────────────────────
TAG_POSITION_IN_BASE = np.array([0.50, 0.005, 0.02])

# ── viz ──────────────────────────────────────────────────────────────────────
AXIS_LENGTH = 0.15

# =============================================================================


def get_intrinsics_from_pipeline(pipeline):
    """Pull actual intrinsics from the active RealSense stream."""
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ])
    dist_coeffs = np.array(intr.coeffs)
    return camera_matrix, dist_coeffs


def capture_frames_and_intrinsics(serial, n_frames=N_FRAMES):
    """Capture frames and pull intrinsics from the live pipeline."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    camera_matrix, dist_coeffs = get_intrinsics_from_pipeline(pipeline)

    frames = []
    for _ in range(n_frames):
        frameset = pipeline.wait_for_frames()
        color_frame = frameset.get_color_frame()
        if color_frame:
            frames.append(np.asarray(color_frame.get_data()).copy())

    pipeline.stop()
    return frames, camera_matrix, dist_coeffs


def detect_tag_pose(image, detector, camera_matrix, dist_coeffs):
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None, None
    for i, tag_id in enumerate(ids.flatten()):
        if tag_id != TAG_ID:
            continue
        s = TAG_SIZE_M / 2.0
        obj_pts = np.array([
            [-s,  s, 0], [ s,  s, 0],
            [ s, -s, 0], [-s, -s, 0],
        ], dtype=np.float32)
        img_pts = corners[i].reshape(4, 2).astype(np.float32)
        ret, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ret:
            return None, None
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.flatten()
        return T, img_pts
    return None, None


def detect_tag_pose_averaged(frames, detector, camera_matrix, dist_coeffs):
    translations, rotations, last_pts = [], [], None
    for frame in frames:
        T, pts = detect_tag_pose(frame, detector, camera_matrix, dist_coeffs)
        if T is not None:
            translations.append(T[:3, 3])
            rotations.append(T[:3, :3])
            last_pts = pts
    if not translations:
        return None, None
    avg_t = np.mean(translations, axis=0)
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    avg_R = U @ Vt
    if np.linalg.det(avg_R) < 0:
        U[:, -1] *= -1
        avg_R = U @ Vt
    T_avg = np.eye(4)
    T_avg[:3, :3] = avg_R
    T_avg[:3, 3] = avg_t
    print(f"  Detected in {len(translations)}/{len(frames)} frames")
    return T_avg, last_pts


def make_T_base_tag(angle_deg):
    theta = np.radians(angle_deg)
    R_bt = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    T = np.eye(4)
    T[:3, :3] = R_bt
    T[:3, 3] = TAG_POSITION_IN_BASE
    return T


def compute_extrinsics(T_cam_tag, angle_deg):
    T_base_tag = make_T_base_tag(angle_deg)
    T_cam_base = T_cam_tag @ np.linalg.inv(T_base_tag)
    return T_cam_base


def draw_axes(frame, ext, camera_matrix, dist_coeffs, label=""):
    axis_pts = np.array([
        [0, 0, 0], [AXIS_LENGTH, 0, 0],
        [0, AXIS_LENGTH, 0], [0, 0, AXIS_LENGTH],
    ], dtype=np.float32)
    rvec, _ = cv2.Rodrigues(ext[:3, :3])
    tvec = ext[:3, 3].reshape(3, 1)
    pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_matrix, dist_coeffs)
    pts = pts.reshape(-1, 2).astype(int)
    vis = frame.copy()
    o, x, y, z = tuple(pts[0]), tuple(pts[1]), tuple(pts[2]), tuple(pts[3])
    cv2.arrowedLine(vis, o, x, (0, 0, 255), 3, tipLength=0.15)
    cv2.arrowedLine(vis, o, y, (0, 255, 0), 3, tipLength=0.15)
    cv2.arrowedLine(vis, o, z, (255, 0, 0), 3, tipLength=0.15)
    cv2.putText(vis, "X", x, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis, "Y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, "Z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    if label:
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return vis



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=int, default=None,
                        help="Lock in this rotation (degrees) for both cameras")
    parser.add_argument("--angle1", type=int, default=None,
                        help="Lock in rotation for cam_1")
    parser.add_argument("--angle2", type=int, default=None,
                        help="Lock in rotation for cam_2")
    args = parser.parse_args()

    locked_angles = {}
    if args.angle is not None:
        locked_angles = {1: args.angle, 2: args.angle}
    if args.angle1 is not None:
        locked_angles[1] = args.angle1
    if args.angle2 is not None:
        locked_angles[2] = args.angle2

    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    aruco_dict = aruco.getPredefinedDictionary(TAG_FAMILY)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    T_ci_b = {}
    cam_frames = {}
    cam_intrinsics = {}

    for cam_id in CAM_IDS:
        key = f"cam_{cam_id}"
        print(f"\n--- Camera {cam_id} (serial {CAM_SERIALS[cam_id]}) ---")

        frames, camera_matrix, dist_coeffs = capture_frames_and_intrinsics(
            CAM_SERIALS[cam_id]
        )
        cam_intrinsics[key] = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
        }

        print(f"  Intrinsics (from device):")
        print(f"    fx={camera_matrix[0,0]:.2f}  fy={camera_matrix[1,1]:.2f}  "
              f"cx={camera_matrix[0,2]:.2f}  cy={camera_matrix[1,2]:.2f}")
        print(f"    dist_coeffs={dist_coeffs.round(6)}")

        T_cam_tag, img_pts = detect_tag_pose_averaged(
            frames, detector, camera_matrix, dist_coeffs
        )
        if T_cam_tag is None:
            print(f"  ERROR: Tag not detected. Skipping.")
            continue

        print(f"  Tag in cam: t={T_cam_tag[:3, 3].round(4)}")
        frame = frames[-1]
        cam_frames[cam_id] = frame

        if cam_id in locked_angles:
            angle = locked_angles[cam_id]
            ext = compute_extrinsics(T_cam_tag, angle)
            cam_in_base = np.linalg.inv(ext)[:3, 3]
            print(f"  Locked angle: {angle}°")
            print(f"  Camera in base: {cam_in_base.round(4)}")
            T_ci_b[key] = ext

            vis = draw_axes(frame, ext, camera_matrix, dist_coeffs,
                            f"cam_{cam_id} @ {angle} deg")
            cv2.imwrite(str(SAVE_DIR / f"cam_{cam_id}_base_axes.png"), vis)
        else:
            print(f"  Generating 4 rotation options:")
            for angle_deg in [0, 90, 180, 270]:
                ext = compute_extrinsics(T_cam_tag, angle_deg)
                cam_in_base = np.linalg.inv(ext)[:3, 3]
                print(f"    {angle_deg:3d}°: cam_in_base = {cam_in_base.round(3)}")

                vis = draw_axes(frame, ext, camera_matrix, dist_coeffs,
                                f"cam_{cam_id} @ {angle_deg} deg")
                save_path = SAVE_DIR / f"cam_{cam_id}_option_{angle_deg}.png"
                cv2.imwrite(str(save_path), vis)

            print(f"\n  Saved 4 images to {SAVE_DIR}/cam_{cam_id}_option_*.png")
            print(f"  → Pick the one where origin is at the robot base & X points forward")
            print(f"  → Then re-run with: --angle{cam_id} <degrees>")

    if locked_angles and len(T_ci_b) > 0:
        calibration_dict = {}
        for cam_id in CAM_IDS:
            key = f"cam_{cam_id}"
            if key not in T_ci_b:
                continue
            calibration_dict[key] = {
                "int": cam_intrinsics[key]["camera_matrix"],
                "dist_coeff": cam_intrinsics[key]["dist_coeffs"],
                "ext": T_ci_b[key],
            }
        np.save(PATH_SAVE_CALIB, calibration_dict)
        print(f"\nSaved calibration to {PATH_SAVE_CALIB}")
        print(f"Cameras calibrated: {list(calibration_dict.keys())}")
    elif not locked_angles:
        print(f"\n{'='*60}")
        print(f"Check the images in {SAVE_DIR}/")
        print(f"Then re-run with the correct angles, e.g.:")
        print(f"  python calibrate_apriltag_table.py --angle1 90 --angle2 270")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()