import torch
import numpy as np

camera2pixelkey = {
    "cam_1": "pixels1",
    "cam_2": "pixels2",
    "cam_51": "pixels51",
}
pixelkey2camera = {v: k for k, v in camera2pixelkey.items()}


def pixel2d_to_3d_torch(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    intrinsic_matrix = torch.tensor(intrinsic_matrix).float().to(depths.device)
    extrinsic_matrix = torch.tensor(extrinsic_matrix).float().to(depths.device)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points3d = torch.stack((x * depths, y * depths, depths), dim=1)  # in camera frame
    points3d = torch.cat(
        (points3d, torch.ones((len(points2d), 1)).to(depths.device)), dim=1
    )
    points3d = (torch.linalg.inv(extrinsic_matrix) @ points3d.T).T  # world frame
    return points3d[..., :3]


def pixel2d_to_3d(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    points2d = np.array(points2d)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points_3d = np.column_stack((x * depths, y * depths, depths))  # in camera frame
    points_3d = np.concatenate([points_3d, np.ones((len(points2d), 1))], axis=1)
    points_3d = (np.linalg.inv(extrinsic_matrix) @ points_3d.T).T  # world frame
    return points_3d[..., :3]


def pixel3d_to_2d(points3d, intrinsic_matrix, camera_projection_matrix):
    points3d = np.array(points3d)
    points3d = np.concatenate([points3d, np.ones((len(points3d), 1))], axis=1)
    points3d = (camera_projection_matrix @ points3d.T).T  # camera frame
    depth = points3d[:, 2]
    points2d = (intrinsic_matrix @ points3d.T).T
    points2d = points2d / points2d[:, 2][:, None]
    return points2d[..., :2], depth


def triangulate_points(P, points):
    """
    Triangulate a batch of points from a variable number of camera views.

    Parameters:
    P: list of 3x4 projection matrices for each camera (currently world2camera transform)
    points: list of Nx2 arrays of normalized image coordinates for each camera

    Returns:
    Nx4 array of homogeneous 3D points
    """
    num_views = len(P)
    assert num_views > 1, "At least 2 cameras are required for triangulation"

    num_points = points[0].shape[0]
    A = np.zeros((num_points, num_views * 2, 4))

    for idx in range(num_views):
        # Set up the linear system for each point
        A[:, idx * 2] = points[idx][:, 0, np.newaxis] * P[idx][2] - P[idx][0]
        A[:, idx * 2 + 1] = points[idx][:, 1, np.newaxis] * P[idx][2] - P[idx][1]

    # Solve the system using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]

    # Normalize the homogeneous coordinates
    X = X / X[:, 3:]

    return X


import numpy as np
from pathlib import Path

def rigid_transform_3D(A, B, ctx=None, dump_dir=None, dump_limit=50, reflect_log=None):
    """
    A, B: Nx3
    ctx: dict with anything you want (demo_idx, obs_idx, cam, frame_idx, etc.)
    dump_dir: if provided, dumps problematic A/B pairs when reflection happens
    reflect_log: list to append reflection events for summary
    """
    assert A.shape == B.shape
    if A.shape[1] != 3:
        raise Exception(f"matrix A is not Nx3, it is {A.shape}")
    if B.shape[1] != 3:
        raise Exception(f"matrix B is not Nx3, it is {B.shape}")

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    U, S, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    detR = np.linalg.det(Rm)

    # Useful “degeneracy” signals:
    # covariance of A, small spread => almost collinear/planar => unstable rotation
    covA = np.cov(Am.T)
    evalsA = np.sort(np.linalg.eigvalsh(covA))  # ascending
    spread = float(evalsA[-1] - evalsA[0])

    # fit error after rotation (before translation)
    # (not perfect metric, but helpful for ranking)
    err = float(np.mean(np.linalg.norm((Am @ Rm.T) - Bm, axis=1)))

    if detR < 0:
        # log with context
        prefix = "[REFLECT]"
        if ctx:
            ctx_str = " ".join([f"{k}={v}" for k, v in ctx.items()])
            prefix = f"{prefix} {ctx_str}"

        print(f"{prefix} det={detR:+.3f} S={S} evalsA={evalsA} spread={spread:.2e} err={err:.4f}")

        # store event for summary
        if reflect_log is not None:
            reflect_log.append({
                "ctx": dict(ctx) if ctx else {},
                "det": float(detR),
                "S": S.copy(),
                "evalsA": evalsA.copy(),
                "spread": spread,
                "err": err,
            })

        # optionally dump the raw point clouds for later
        if dump_dir is not None and (reflect_log is None or len(reflect_log) <= dump_limit):
            dump_dir = Path(dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)

            tag = "unknown"
            if ctx:
                tag = "_".join([f"{k}{v}" for k, v in ctx.items()])
            np.savez_compressed(dump_dir / f"reflect_{tag}.npz", A=A, B=B, S=S, evalsA=evalsA, spread=spread, err=err)

        # correct reflection
        Vt[2, :] *= -1
        Rm = Vt.T @ U.T

    t = -Rm @ centroid_A.T + centroid_B.T
    return Rm, t


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation to rotation matrix
    using Gram-Schmidt orthogonalization.

    Args:
        d6: 6D rotation representation, of shape (..., 6)

    Returns:
        Batch of rotation matrices of shape (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation
    by dropping the last row.

    Args:
        matrix: Batch of rotation matrices of shape (..., 3, 3)

    Returns:
        6D rotation representation, of shape (..., 6)
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))
