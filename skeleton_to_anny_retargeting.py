#!/usr/bin/env python3
"""
Retargeting module to fit ANNY model to biomechanical skeleton poses.

This module handles:
1. Coordinate frame transformation between ISB and ANNY conventions
2. Neutral pose offset compensation
3. Joint angle remapping
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
from typing import Dict, Tuple, Optional


class SkeletonToAnnyRetargeter:
    """
    Retargets joint angles from biomechanical skeleton (ISB convention) 
    to ANNY model coordinate system.
    
    ISB Convention (Skeleton):
    - Y: Longitudinal to segment (upward)
    - Z: Medio-lateral (middle to right)
    - X: Antero-posterior (back to front)
    
    ANNY Convention:
    - Z: Upward
    - X: Medio-lateral
    - Y: Postero-anterior
    """
    
    def __init__(self, skeleton_env, anny_env):
        """
        Initialize the retargeter with both environments.
        
        Args:
            skeleton_env: The SkeletonTorque environment
            anny_env: The ANNY environment
        """
        self.skeleton_env = skeleton_env
        self.anny_env = anny_env
        
        # Store neutral poses
        self.q_neutral_skeleton = None
        self.q_neutral_anny = None
        
        # Compute frame transformation matrix (ISB to ANNY)
        self.frame_transform = self._compute_frame_transformation()
        
        # Joint mapping between models (may need manual configuration)
        self.joint_mapping = self._initialize_joint_mapping()
        
        # Calibrate neutral poses
        self._calibrate_neutral_poses()
    
    def _compute_frame_transformation(self) -> np.ndarray:
        """
        Compute the rotation matrix to transform from ISB to ANNY convention.
        
        ISB to ANNY transformation:
        ISB_Y -> ANNY_Z (upward)
        ISB_Z -> ANNY_X (medio-lateral) 
        ISB_X -> ANNY_Y (but flipped: antero-posterior to postero-anterior)
        
        Returns:
            3x3 rotation matrix for frame transformation
        """
        # This transforms from ISB axes to ANNY axes
        # Row i tells where ISB axis i goes in ANNY frame
        transform = np.array([
            [0, 1, 0],   # ISB_X -> ANNY_Y
            [0, 0, 1],   # ISB_Y -> ANNY_Z
            [1, 0, 0]    # ISB_Z -> ANNY_X
        ])
        
        # Account for the flip in anterior-posterior direction
        # ISB_X is antero-posterior, ANNY_Y is postero-anterior (opposite)
        transform[0, :] *= -1
        
        return transform
    
    def _initialize_joint_mapping(self) -> Dict[str, str]:
        """
        Initialize mapping between skeleton joint names and ANNY joint names.
        
        This is a placeholder - you'll need to fill in actual joint mappings
        based on the model definitions.
        
        Returns:
            Dictionary mapping skeleton joint names to ANNY joint names
        """
        # Example mapping - adjust based on actual joint names
        mapping = {
            # Spine
            'lumbar': 'abdomen',
            'thorax': 'chest',
            
            # Arms
            'shoulder_r': 'r_shoulder',
            'elbow_r': 'r_elbow',
            'wrist_r': 'r_wrist',
            'shoulder_l': 'l_shoulder',
            'elbow_l': 'l_elbow',
            'wrist_l': 'l_wrist',
            
            # Legs  
            'hip_r': 'r_hip',
            'knee_r': 'r_knee',
            'ankle_r': 'r_ankle',
            'hip_l': 'l_hip',
            'knee_l': 'l_knee',
            'ankle_l': 'l_ankle',
            
            # Head/Neck
            'neck': 'neck',
            'head': 'head'
        }
        
        return mapping
    
    def _calibrate_neutral_poses(self):
        """
        Calibrate the neutral poses of both models.
        """
        # Reset both environments to get neutral poses
        self.skeleton_env.reset()
        self.q_neutral_skeleton = self.skeleton_env._data.qpos.copy()
        
        self.anny_env.reset()
        self.q_neutral_anny = self.anny_env._data.qpos.copy()
        
        print(f"Skeleton neutral pose shape: {self.q_neutral_skeleton.shape}")
        print(f"ANNY neutral pose shape: {self.q_neutral_anny.shape}")
    
    def transform_joint_angles(self, q_skeleton: np.ndarray) -> np.ndarray:
        """
        Transform joint angles from skeleton to ANNY model.
        
        Args:
            q_skeleton: Joint angles from the skeleton model
            
        Returns:
            q_anny: Transformed joint angles for ANNY model
        """
        # Start with ANNY neutral pose
        q_anny = self.q_neutral_anny.copy()
        
        # Compute deviation from skeleton neutral
        dq_skeleton = q_skeleton - self.q_neutral_skeleton
        
        # For each joint, transform the angular deviation
        for skel_joint, anny_joint in self.joint_mapping.items():
            skel_idx = self._get_joint_qpos_indices(self.skeleton_env, skel_joint)
            anny_idx = self._get_joint_qpos_indices(self.anny_env, anny_joint)
            
            if skel_idx is None or anny_idx is None:
                continue
            
            # Get the joint angle deviation
            joint_angles = dq_skeleton[skel_idx]
            
            # Transform based on joint type
            if len(joint_angles) == 3:  # Ball joint (3 DOF)
                transformed = self._transform_ball_joint(joint_angles)
            elif len(joint_angles) == 1:  # Hinge joint (1 DOF)
                transformed = self._transform_hinge_joint(joint_angles, skel_joint)
            else:  # 2 DOF or other
                transformed = joint_angles  # Direct copy for now
            
            # Apply to ANNY model (add to neutral)
            q_anny[anny_idx] += transformed
        
        return q_anny
    
    def _transform_ball_joint(self, angles: np.ndarray) -> np.ndarray:
        """
        Transform 3-DOF ball joint angles between coordinate frames.
        
        Args:
            angles: 3D rotation angles in ISB frame
            
        Returns:
            Transformed angles in ANNY frame
        """
        # Convert angles to rotation matrix
        rot_isb = R.from_euler('xyz', angles)
        
        # Transform to ANNY frame
        # R_anny = T @ R_isb @ T^T (similarity transform)
        rot_matrix_isb = rot_isb.as_matrix()
        rot_matrix_anny = self.frame_transform @ rot_matrix_isb @ self.frame_transform.T
        
        # Convert back to Euler angles
        rot_anny = R.from_matrix(rot_matrix_anny)
        angles_anny = rot_anny.as_euler('xyz')
        
        return angles_anny
    
    def _transform_hinge_joint(self, angle: np.ndarray, joint_name: str) -> np.ndarray:
        """
        Transform 1-DOF hinge joint angle.
        
        Args:
            angle: Single rotation angle
            joint_name: Name of the joint (to determine axis)
            
        Returns:
            Transformed angle
        """
        # For hinge joints, we need to know the rotation axis
        # This is model-specific and may need manual configuration
        
        # Simple sign flip may be needed depending on axis alignment
        # This is a placeholder - adjust based on actual joint definitions
        if 'elbow' in joint_name or 'knee' in joint_name:
            # These typically need sign flip between conventions
            return -angle
        
        return angle
    
    def _get_joint_qpos_indices(self, env, joint_name: str) -> Optional[np.ndarray]:
        """
        Get the qpos indices for a given joint name.
        
        Args:
            env: The MuJoCo environment
            joint_name: Name of the joint
            
        Returns:
            Array of qpos indices for this joint, or None if not found
        """
        try:
            joint_id = env._model.joint(joint_name).id
            qpos_start = env._model.jnt_qposadr[joint_id]
            joint_type = env._model.jnt_type[joint_id]
            
            # Determine number of DOFs based on joint type
            if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                n_dof = 4  # Quaternion representation
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                n_dof = 1
            elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                n_dof = 1
            else:
                n_dof = 1  # Default
            
            return np.arange(qpos_start, qpos_start + n_dof)
        except:
            return None
    
    def retarget_trajectory(self, skeleton_trajectory: np.ndarray) -> np.ndarray:
        """
        Retarget an entire trajectory from skeleton to ANNY.
        
        Args:
            skeleton_trajectory: Tx N array of skeleton joint angles over time
            
        Returns:
            anny_trajectory: Tx M array of ANNY joint angles
        """
        n_timesteps = skeleton_trajectory.shape[0]
        n_anny_joints = len(self.q_neutral_anny)
        
        anny_trajectory = np.zeros((n_timesteps, n_anny_joints))
        
        for t in range(n_timesteps):
            anny_trajectory[t] = self.transform_joint_angles(skeleton_trajectory[t])
        
        return anny_trajectory
    
    def visualize_comparison(self, q_skeleton: np.ndarray):
        """
        Visualize both models side by side with the same pose.
        
        Args:
            q_skeleton: Joint angles for the skeleton model
        """
        # Transform to ANNY
        q_anny = self.transform_joint_angles(q_skeleton)
        
        # Set poses
        self.skeleton_env._data.qpos[:] = q_skeleton
        mujoco.mj_forward(self.skeleton_env._model, self.skeleton_env._data)
        
        self.anny_env._data.qpos[:] = q_anny
        mujoco.mj_forward(self.anny_env._model, self.anny_env._data)
        
        print("Skeleton pose set. ANNY pose transformed and set.")
        print(f"Skeleton qpos: {q_skeleton[:10]}...")  # Show first 10 values
        print(f"ANNY qpos: {q_anny[:10]}...")


def calibrate_with_matching_poses(retargeter, skeleton_poses, anny_poses):
    """
    Fine-tune the retargeting using known corresponding poses.
    
    Args:
        retargeter: The SkeletonToAnnyRetargeter instance
        skeleton_poses: List of skeleton poses
        anny_poses: List of corresponding ANNY poses
    """
    # This could be used to learn better joint mappings or offsets
    # using optimization techniques
    pass


if __name__ == "__main__":
    # Example usage
    from loco_mujoco.environments.humanoids import SkeletonTorque
    # Assuming you have an ANNY environment class
    # from your_module import ANNY
    
    print("Creating retargeter for skeleton to ANNY transformation...")
    
    # Create environments
    skeleton_env = SkeletonTorque()
    # anny_env = ANNY()  # You'll need to import/create this
    
    # Create retargeter
    # retargeter = SkeletonToAnnyRetargeter(skeleton_env, anny_env)
    
    # Example: Transform a single pose
    # skeleton_env.reset()
    # q_skeleton = skeleton_env._data.qpos.copy()
    # q_anny = retargeter.transform_joint_angles(q_skeleton)
    
    print("Retargeting system ready.")
