#!/usr/bin/env python3
"""
Complete retargeting pipeline: Skeleton to ANNY model fitting.

This script demonstrates the full pipeline for fitting the ANNY model
to biomechanical skeleton poses, handling both coordinate frame
differences and neutral pose offsets.
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import time
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CompleteRetargetingPipeline:
    """
    Complete pipeline for skeleton to ANNY retargeting.
    """
    
    def __init__(self, skeleton_env, anny_env):
        """
        Initialize the complete retargeting pipeline.
        
        Args:
            skeleton_env: SkeletonTorque environment
            anny_env: ANNY environment
        """
        self.skeleton_env = skeleton_env
        self.anny_env = anny_env
        
        # Frame transformation matrices
        self.setup_frame_transformations()
        
        # Joint correspondence
        self.setup_joint_mapping()
        
        # Calibration data
        self.calibration_data = {
            'q_offset': None,  # Neutral pose offset
            'scale_factors': None,  # Per-joint scaling
            'rotation_offsets': {}  # Per-joint rotation offsets
        }
    
    def setup_frame_transformations(self):
        """
        Setup coordinate frame transformations between ISB and ANNY.
        """
        # ISB to ANNY coordinate transformation
        # ISB: Y-up (longitudinal), Z-right (medio-lateral), X-front (antero-posterior)  
        # ANNY: Z-up, X-right (medio-lateral), Y-back (postero-anterior)
        
        self.isb_to_anny = np.array([
            [0, 0, 1],   # ISB-X (front) -> ANNY-Z (up) - NO, this is wrong
            [0, -1, 0],  # ISB-Y (up) -> -ANNY-Y (back) 
            [1, 0, 0]    # ISB-Z (right) -> ANNY-X (right)
        ])
        
        # Correct transformation based on your description:
        # ISB-Y (up) -> ANNY-Z (up)
        # ISB-Z (right) -> ANNY-X (right)
        # ISB-X (front) -> -ANNY-Y (since ANNY-Y is back)
        self.isb_to_anny = np.array([
            [0, -1, 0],  # ISB-X -> -ANNY-Y
            [0, 0, 1],   # ISB-Y -> ANNY-Z
            [1, 0, 0]    # ISB-Z -> ANNY-X
        ])
        
        self.anny_to_isb = self.isb_to_anny.T
        
        print("Frame transformations initialized:")
        print("ISB to ANNY transformation matrix:")
        print(self.isb_to_anny)
    
    def setup_joint_mapping(self):
        """
        Setup joint name mapping and DOF correspondence.
        """
        # This mapping needs to be adjusted based on actual model joints
        self.joint_map = {}
        
        # Try to auto-detect joint correspondence based on naming patterns
        skeleton_joints = self._get_all_joint_names(self.skeleton_env)
        anny_joints = self._get_all_joint_names(self.anny_env)
        
        print(f"\nSkeleton joints ({len(skeleton_joints)}): {skeleton_joints[:5]}...")
        print(f"ANNY joints ({len(anny_joints)}): {anny_joints[:5]}...")
        
        # Common patterns for joint matching
        patterns = [
            ('hip', 'hip'),
            ('knee', 'knee'),
            ('ankle', 'ankle'),
            ('shoulder', 'shoulder'),
            ('elbow', 'elbow'),
            ('wrist', 'wrist'),
            ('spine', 'spine'),
            ('lumbar', 'abdomen'),
            ('thorax', 'chest'),
            ('neck', 'neck'),
            ('head', 'head')
        ]
        
        for skel_joint in skeleton_joints:
            for anny_joint in anny_joints:
                for skel_pattern, anny_pattern in patterns:
                    if skel_pattern in skel_joint.lower() and anny_pattern in anny_joint.lower():
                        # Check for side correspondence (left/right)
                        if ('_r' in skel_joint and 'right' in anny_joint) or \
                           ('_l' in skel_joint and 'left' in anny_joint) or \
                           ('right' in skel_joint and 'right' in anny_joint) or \
                           ('left' in skel_joint and 'left' in anny_joint) or \
                           ('_r' not in skel_joint and '_l' not in skel_joint and 
                            'right' not in anny_joint and 'left' not in anny_joint):
                            self.joint_map[skel_joint] = anny_joint
                            break
        
        print(f"\nAuto-detected {len(self.joint_map)} joint mappings")
        for k, v in list(self.joint_map.items())[:5]:
            print(f"  {k} -> {v}")
    
    def _get_all_joint_names(self, env) -> list:
        """Get all joint names from environment."""
        names = []
        for i in range(env._model.njnt):
            name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                names.append(name)
        return names
    
    def calibrate_neutral_poses(self):
        """
        Calibrate the neutral pose offset between models.
        """
        print("\nCalibrating neutral poses...")
        
        # Reset both to neutral
        self.skeleton_env.reset()
        q_neutral_skeleton = self.skeleton_env._data.qpos.copy()
        
        self.anny_env.reset()  
        q_neutral_anny = self.anny_env._data.qpos.copy()
        
        # Store dimensions
        self.n_skeleton_dof = len(q_neutral_skeleton)
        self.n_anny_dof = len(q_neutral_anny)
        
        print(f"Skeleton DOF: {self.n_skeleton_dof}")
        print(f"ANNY DOF: {self.n_anny_dof}")
        
        # Compute initial offset (simplified - assumes similar ordering)
        min_dof = min(self.n_skeleton_dof, self.n_anny_dof)
        self.calibration_data['q_offset'] = q_neutral_anny[:min_dof] - q_neutral_skeleton[:min_dof]
        self.calibration_data['scale_factors'] = np.ones(min_dof)
        
        return q_neutral_skeleton, q_neutral_anny
    
    def transform_single_pose(self, q_skeleton: np.ndarray, 
                            use_optimization: bool = False) -> np.ndarray:
        """
        Transform a single skeleton pose to ANNY.
        
        Args:
            q_skeleton: Skeleton joint angles
            use_optimization: Whether to use optimization for better fit
            
        Returns:
            q_anny: Transformed ANNY joint angles
        """
        # Initialize with ANNY neutral pose
        q_anny = np.zeros(self.n_anny_dof)
        
        # Get neutral poses if not calibrated
        if self.calibration_data['q_offset'] is None:
            self.calibrate_neutral_poses()
        
        # Simple approach: direct mapping with offset and scaling
        min_dof = min(len(q_skeleton), len(q_anny))
        
        for i in range(min_dof):
            # Apply offset and scaling
            q_anny[i] = (q_skeleton[i] + self.calibration_data['q_offset'][i]) * \
                       self.calibration_data['scale_factors'][i]
        
        # Advanced: Handle specific joints with frame transformation
        for skel_joint, anny_joint in self.joint_map.items():
            skel_idx = self._get_joint_indices(self.skeleton_env, skel_joint)
            anny_idx = self._get_joint_indices(self.anny_env, anny_joint)
            
            if skel_idx is not None and anny_idx is not None:
                # Get joint type
                joint_type = self._get_joint_type(self.skeleton_env, skel_joint)
                
                if joint_type == 'ball':  # 3 DOF rotation
                    # Apply coordinate transformation for ball joints
                    angles_isb = q_skeleton[skel_idx]
                    angles_anny = self._transform_rotation(angles_isb)
                    q_anny[anny_idx] = angles_anny
                elif joint_type == 'hinge':  # 1 DOF
                    # Simple scaling for hinge joints
                    if len(skel_idx) == 1 and len(anny_idx) == 1:
                        q_anny[anny_idx[0]] = q_skeleton[skel_idx[0]] * \
                                             self.calibration_data['scale_factors'][skel_idx[0]]
        
        if use_optimization:
            # Fine-tune with optimization
            q_anny = self._optimize_pose(q_skeleton, q_anny)
        
        return q_anny
    
    def _transform_rotation(self, angles_isb: np.ndarray) -> np.ndarray:
        """
        Transform rotation angles from ISB to ANNY frame.
        """
        if len(angles_isb) == 3:
            # Convert to rotation matrix
            rot_isb = R.from_euler('xyz', angles_isb)
            mat_isb = rot_isb.as_matrix()
            
            # Apply coordinate transformation
            mat_anny = self.isb_to_anny @ mat_isb @ self.isb_to_anny.T
            
            # Convert back to Euler angles
            rot_anny = R.from_matrix(mat_anny)
            return rot_anny.as_euler('xyz')
        elif len(angles_isb) == 4:
            # Quaternion
            rot_isb = R.from_quat(angles_isb)
            mat_isb = rot_isb.as_matrix()
            
            # Apply coordinate transformation
            mat_anny = self.isb_to_anny @ mat_isb @ self.isb_to_anny.T
            
            # Convert back to quaternion
            rot_anny = R.from_matrix(mat_anny)
            return rot_anny.as_quat()
        else:
            return angles_isb
    
    def _get_joint_indices(self, env, joint_name: str) -> Optional[np.ndarray]:
        """Get qpos indices for a joint."""
        try:
            joint_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                return None
            
            qpos_start = env._model.jnt_qposadr[joint_id]
            joint_type = env._model.jnt_type[joint_id]
            
            # Get DOF count based on type
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                n_dof = 7  # 3 trans + 4 quat
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                n_dof = 4  # quaternion
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                n_dof = 1
            elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                n_dof = 1
            else:
                n_dof = 1
            
            return np.arange(qpos_start, qpos_start + n_dof)
        except:
            return None
    
    def _get_joint_type(self, env, joint_name: str) -> str:
        """Get joint type as string."""
        try:
            joint_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_type = env._model.jnt_type[joint_id]
            
            if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                return 'ball'
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                return 'hinge'
            elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                return 'slide'
            else:
                return 'other'
        except:
            return 'unknown'
    
    def _optimize_pose(self, q_skeleton: np.ndarray, q_anny_init: np.ndarray) -> np.ndarray:
        """
        Optimize ANNY pose to better match skeleton configuration.
        """
        # This would implement IK or optimization to minimize
        # the difference in end-effector positions
        # For now, return the initial guess
        return q_anny_init
    
    def transform_trajectory(self, skeleton_trajectory: np.ndarray,
                           show_progress: bool = True) -> np.ndarray:
        """
        Transform entire trajectory from skeleton to ANNY.
        
        Args:
            skeleton_trajectory: T x N array of skeleton poses
            show_progress: Whether to show progress bar
            
        Returns:
            anny_trajectory: T x M array of ANNY poses
        """
        n_frames = len(skeleton_trajectory)
        anny_trajectory = np.zeros((n_frames, self.n_anny_dof))
        
        print(f"\nTransforming {n_frames} frames...")
        
        for i in range(n_frames):
            if show_progress and i % 10 == 0:
                print(f"  Frame {i}/{n_frames}")
            
            anny_trajectory[i] = self.transform_single_pose(skeleton_trajectory[i])
        
        print("Transformation complete!")
        return anny_trajectory
    
    def visualize_comparison(self, q_skeleton: np.ndarray, 
                            q_anny: Optional[np.ndarray] = None):
        """
        Visualize skeleton and ANNY models side by side.
        
        Args:
            q_skeleton: Skeleton pose
            q_anny: ANNY pose (if None, will be computed)
        """
        # Set skeleton pose
        self.skeleton_env._data.qpos[:] = q_skeleton
        mujoco.mj_forward(self.skeleton_env._model, self.skeleton_env._data)
        
        # Compute or use provided ANNY pose
        if q_anny is None:
            q_anny = self.transform_single_pose(q_skeleton)
        
        # Set ANNY pose
        self.anny_env._data.qpos[:] = q_anny
        mujoco.mj_forward(self.anny_env._model, self.anny_env._data)
        
        print("Poses set for visualization")
        
        # Render both
        self.skeleton_env.render()
        self.anny_env.render()
    
    def validate_retargeting(self, skeleton_trajectory: np.ndarray) -> dict:
        """
        Validate retargeting quality using various metrics.
        
        Args:
            skeleton_trajectory: Trajectory to validate
            
        Returns:
            Dictionary of validation metrics
        """
        print("\nValidating retargeting quality...")
        
        anny_trajectory = self.transform_trajectory(skeleton_trajectory, show_progress=False)
        
        metrics = {
            'joint_range': [],
            'smoothness': [],
            'marker_error': []
        }
        
        # Check joint ranges
        for i in range(self.n_anny_dof):
            joint_range = np.max(anny_trajectory[:, i]) - np.min(anny_trajectory[:, i])
            metrics['joint_range'].append(joint_range)
        
        # Check smoothness (velocity)
        velocity = np.diff(anny_trajectory, axis=0)
        metrics['smoothness'] = np.mean(np.abs(velocity), axis=0)
        
        # Check marker correspondence (if available)
        # This would compare anatomical landmark positions
        
        print(f"Average joint range: {np.mean(metrics['joint_range']):.3f} rad")
        print(f"Average smoothness: {np.mean(metrics['smoothness']):.3f} rad/frame")
        
        return metrics


def main():
    """
    Main demonstration of the retargeting pipeline.
    """
    print("=" * 60)
    print("SKELETON TO ANNY RETARGETING PIPELINE")
    print("=" * 60)
    
    from loco_mujoco.environments.humanoids import SkeletonTorque
    from loco_mujoco.task_factories import ImitationFactory
    from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf
    
    # Create skeleton environment
    print("\n1. Loading skeleton environment...")
    skeleton_env = SkeletonTorque()
    
    # Create ANNY environment (you need to import this)
    print("\n2. Loading ANNY environment...")
    # from your_module import ANNY
    # anny_env = ANNY()
    
    # For demonstration, we'll use skeleton_env as placeholder
    anny_env = skeleton_env  # REPLACE with actual ANNY environment
    
    # Create retargeting pipeline
    print("\n3. Initializing retargeting pipeline...")
    pipeline = CompleteRetargetingPipeline(skeleton_env, anny_env)
    
    # Calibrate neutral poses
    print("\n4. Calibrating neutral poses...")
    q_neutral_skel, q_neutral_anny = pipeline.calibrate_neutral_poses()
    
    # Load squat trajectory
    print("\n5. Loading squat trajectory...")
    dataset_conf = DefaultDatasetConf(task=["squat"], dataset_type="mocap")
    trajectory = ImitationFactory.get_default_traj(skeleton_env, dataset_conf)
    skeleton_trajectory = np.array(trajectory.data.qpos)
    
    print(f"   Loaded {len(skeleton_trajectory)} frames")
    
    # Transform trajectory
    print("\n6. Transforming trajectory to ANNY model...")
    anny_trajectory = pipeline.transform_trajectory(skeleton_trajectory[:100])  # First 100 frames
    
    # Validate
    print("\n7. Validating retargeting quality...")
    metrics = pipeline.validate_retargeting(skeleton_trajectory[:100])
    
    # Visualize a few key frames
    print("\n8. Ready for visualization")
    print("   Use pipeline.visualize_comparison(skeleton_trajectory[i]) to visualize frame i")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return pipeline, skeleton_trajectory, anny_trajectory


if __name__ == "__main__":
    pipeline, skel_traj, anny_traj = main()
    
    print("\nNext steps:")
    print("1. Replace the placeholder ANNY environment with actual ANNY model")
    print("2. Fine-tune joint mappings in setup_joint_mapping()")
    print("3. Add optimization-based refinement if needed")
    print("4. Visualize results with both models side by side")
