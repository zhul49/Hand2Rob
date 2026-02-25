"""
Inverse Kinematics solver for Franka Panda robot using PyBullet.
Validates poses and finds feasible configurations to avoid joint limits.
"""

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R


class FrankaIKSolver:
    def __init__(self, use_gui=False):
        """
        Initialize PyBullet IK solver for Franka Panda.

        Args:
            use_gui: If True, show PyBullet GUI (for debugging)
        """
        # Connect to PyBullet
        if use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load Franka robot
        # PyBullet includes a Franka model in pybullet_data
        try:
            self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        except:
            # Fallback: try loading from common paths
            import os
            possible_paths = [
                "/usr/local/lib/python3.8/dist-packages/pybullet_data/franka_panda/panda.urdf",
                "/usr/share/pybullet_data/franka_panda/panda.urdf",
            ]
            loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.robot_id = p.loadURDF(path, useFixedBase=True)
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError("Could not find Franka URDF file")

        # Franka end-effector link index (link 11 is the flange)
        self.ee_link_index = 11

        # Joint indices for the 7 arm joints (excluding gripper)
        self.arm_joint_indices = list(range(7))

        # Joint limits for Franka Panda
        self.joint_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Set robot to a neutral pose
        neutral_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(neutral_joint_positions):
            p.resetJointState(self.robot_id, i, pos)

        self.current_joint_positions = np.array(neutral_joint_positions)

    def solve_ik(self, target_pos, target_orn_quat, current_joints=None):
        """
        Solve IK for target pose.

        Args:
            target_pos: Target position [x, y, z] in meters
            target_orn_quat: Target orientation as quaternion [x, y, z, w]
            current_joints: Current joint positions (optional, for seeding)

        Returns:
            joint_positions: Array of 7 joint positions, or None if no solution
            is_valid: Boolean indicating if solution is within joint limits
        """
        if current_joints is not None:
            # Set current joint positions as seed
            for i, pos in enumerate(current_joints):
                p.resetJointState(self.robot_id, i, pos)

        # Solve IK
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            target_orn_quat,
            lowerLimits=self.joint_lower_limits.tolist(),
            upperLimits=self.joint_upper_limits.tolist(),
            jointRanges=(self.joint_upper_limits - self.joint_lower_limits).tolist(),
            restPoses=self.current_joint_positions.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5
        )

        joint_positions = np.array(joint_positions[:7])

        # Check if solution is within joint limits (with small margin)
        margin = 0.05  # 0.05 radians margin from limits
        is_valid = np.all(joint_positions >= self.joint_lower_limits + margin) and \
                   np.all(joint_positions <= self.joint_upper_limits - margin)

        return joint_positions, is_valid

    def find_feasible_orientation(self, target_pos, target_orn_quat, current_joints=None, max_attempts=10):
        """
        Find a feasible orientation close to the target that avoids joint limits.

        Args:
            target_pos: Target position [x, y, z]
            target_orn_quat: Target orientation quaternion [x, y, z, w]
            current_joints: Current joint positions
            max_attempts: Maximum number of orientation adjustments to try

        Returns:
            feasible_orn_quat: Feasible orientation quaternion, or original if no better found
            is_valid: Whether a feasible solution was found
        """
        # First try the original orientation
        joint_pos, is_valid = self.solve_ik(target_pos, target_orn_quat, current_joints)

        if is_valid:
            self.current_joint_positions = joint_pos
            return target_orn_quat, True

        # If not valid, try adjusting the orientation
        # Convert to euler angles and try variations
        target_rot = R.from_quat(target_orn_quat)
        euler_xyz = target_rot.as_euler('xyz', degrees=False)

        # Try adjusting pitch (most common issue for horizontal orientations)
        pitch_adjustments = np.linspace(-0.3, 0.3, max_attempts)  # +/- ~17 degrees

        for pitch_adj in pitch_adjustments:
            adjusted_euler = euler_xyz.copy()
            adjusted_euler[1] += pitch_adj  # Adjust pitch

            adjusted_rot = R.from_euler('xyz', adjusted_euler)
            adjusted_quat = adjusted_rot.as_quat()

            joint_pos, is_valid = self.solve_ik(target_pos, adjusted_quat, current_joints)

            if is_valid:
                self.current_joint_positions = joint_pos
                print(f"[IK] Found feasible orientation with pitch adjustment: {np.degrees(pitch_adj):.1f} deg")
                return adjusted_quat, True

        # If still no valid solution, try adjusting roll as well
        for roll_adj in np.linspace(-0.2, 0.2, 5):
            for pitch_adj in np.linspace(-0.3, 0.3, 5):
                adjusted_euler = euler_xyz.copy()
                adjusted_euler[0] += roll_adj   # Adjust roll
                adjusted_euler[1] += pitch_adj  # Adjust pitch

                adjusted_rot = R.from_euler('xyz', adjusted_euler)
                adjusted_quat = adjusted_rot.as_quat()

                joint_pos, is_valid = self.solve_ik(target_pos, adjusted_quat, current_joints)

                if is_valid:
                    self.current_joint_positions = joint_pos
                    print(f"[IK] Found feasible orientation with roll/pitch adjustment: "
                          f"roll={np.degrees(roll_adj):.1f}°, pitch={np.degrees(pitch_adj):.1f}°")
                    return adjusted_quat, True

        # No feasible orientation found, return original
        print("[IK] Warning: Could not find feasible orientation, using original (may hit joint limits)")
        return target_orn_quat, False

    def __del__(self):
        """Cleanup PyBullet connection"""
        try:
            p.disconnect(self.physics_client)
        except:
            pass
