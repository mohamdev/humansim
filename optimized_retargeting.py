#!/usr/bin/env python3
"""
Optimization-based retargeting to learn transformation parameters.

This module uses optimization to find the best transformation between
skeleton and ANNY models, including:
- Per-joint rotation offsets
- Neutral pose calibration
- Joint gain/scaling factors
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation as R
import mujoco
from typing import List, Tuple, Optional, Dict


class OptimizedRetargeter:
    """
    Learn optimal retargeting parameters through optimization.
    """
    
    def __init__(self, skeleton_env, anny_env):
        """
        Initialize the optimization-based retargeter.
        
        Args:
            skeleton_env: The SkeletonTorque environment
            anny_env: The ANNY environment
        """
        self.skeleton_env = skeleton_env
        self.anny_env = anny_env
        
        # Get model dimensions
        self.n_skeleton_joints = len(skeleton_env._data.qpos)
        self.n_anny_joints = len(anny_env._data.qpos)
        
        # Parameters to optimize
        self.params = {
            'neutral_offset': np.zeros(self.n_anny_joints),
            'joint_gains': np.ones(min(self.n_skeleton_joints, self.n_anny_joints)),
            'rotation_offset': np.eye(3),  # Global rotation offset
            'per_joint_rotation': {}  # Per-joint rotation offsets
        }
        
        # Site/marker correspondence for optimization
        self.marker_pairs = self._identify_marker_correspondence()
    
    def _identify_marker_correspondence(self) -> List[Tuple[str, str]]:
        """
        Identify corresponding markers/sites between models.
        
        Returns:
            List of (skeleton_site, anny_site) pairs
        """
        # Common anatomical landmarks
        pairs = [
            ('head', 'head'),
            ('r_shoulder', 'right_shoulder'),
            ('l_shoulder', 'left_shoulder'),
            ('r_elbow', 'right_elbow'),
            ('l_elbow', 'left_elbow'),
            ('r_wrist', 'right_wrist'),
            ('l_wrist', 'left_wrist'),
            ('r_hip', 'right_hip'),
            ('l_hip', 'left_hip'),
            ('r_knee', 'right_knee'),
            ('l_knee', 'left_knee'),
            ('r_ankle', 'right_ankle'),
            ('l_ankle', 'left_ankle'),
            ('pelvis', 'pelvis'),
            ('chest', 'thorax')
        ]
        
        # Filter to only existing sites
        valid_pairs = []
        for skel_site, anny_site in pairs:
            try:
                self.skeleton_env._model.site(skel_site)
                self.anny_env._model.site(anny_site)
                valid_pairs.append((skel_site, anny_site))
            except:
                continue
        
        return valid_pairs
    
    def compute_marker_positions(self, env, site_names: List[str]) -> np.ndarray:
        """
        Get 3D positions of markers/sites.
        
        Args:
            env: MuJoCo environment
            site_names: List of site names
            
        Returns:
            Nx3 array of marker positions
        """
        positions = []
        for site_name in site_names:
            try:
                site_id = env._model.site(site_name).id
                pos = env._data.site_xpos[site_id].copy()
                positions.append(pos)
            except:
                positions.append(np.zeros(3))
        
        return np.array(positions)
    
    def objective_function(self, params: np.ndarray, 
                          skeleton_poses: np.ndarray,
                          anny_poses: np.ndarray) -> float:
        """
        Compute the retargeting error for optimization.
        
        Args:
            params: Flattened parameter vector
            skeleton_poses: Reference skeleton poses
            anny_poses: Target ANNY poses
            
        Returns:
            Total error (sum of position and orientation errors)
        """
        # Unpack parameters
        self._unpack_params(params)
        
        total_error = 0.0
        n_poses = len(skeleton_poses)
        
        for i in range(n_poses):
            # Apply retargeting
            q_anny_pred = self.retarget(skeleton_poses[i])
            
            # Set predicted pose
            self.anny_env._data.qpos[:] = q_anny_pred
            mujoco.mj_forward(self.anny_env._model, self.anny_env._data)
            
            # Set ground truth pose
            self.skeleton_env._data.qpos[:] = skeleton_poses[i]
            mujoco.mj_forward(self.skeleton_env._model, self.skeleton_env._data)
            
            # Compute marker position error
            if self.marker_pairs:
                skel_sites = [p[0] for p in self.marker_pairs]
                anny_sites = [p[1] for p in self.marker_pairs]
                
                skel_pos = self.compute_marker_positions(self.skeleton_env, skel_sites)
                anny_pos = self.compute_marker_positions(self.anny_env, anny_sites)
                
                # Position error (MSE)
                pos_error = np.mean(np.sum((skel_pos - anny_pos)**2, axis=1))
                total_error += pos_error
            
            # Joint angle error (if ground truth available)
            if anny_poses is not None:
                joint_error = np.mean((q_anny_pred - anny_poses[i])**2)
                total_error += 0.1 * joint_error  # Weight joint error less
        
        return total_error / n_poses
    
    def _pack_params(self) -> np.ndarray:
        """Pack parameters into a flat vector for optimization."""
        params_list = []
        
        # Neutral offset
        params_list.append(self.params['neutral_offset'])
        
        # Joint gains
        params_list.append(self.params['joint_gains'])
        
        # Global rotation (as axis-angle)
        rot = R.from_matrix(self.params['rotation_offset'])
        params_list.append(rot.as_rotvec())
        
        return np.concatenate(params_list)
    
    def _unpack_params(self, params: np.ndarray):
        """Unpack flat parameter vector."""
        idx = 0
        
        # Neutral offset
        self.params['neutral_offset'] = params[idx:idx + self.n_anny_joints]
        idx += self.n_anny_joints
        
        # Joint gains
        n_gains = len(self.params['joint_gains'])
        self.params['joint_gains'] = params[idx:idx + n_gains]
        idx += n_gains
        
        # Global rotation
        rotvec = params[idx:idx + 3]
        self.params['rotation_offset'] = R.from_rotvec(rotvec).as_matrix()
    
    def retarget(self, q_skeleton: np.ndarray) -> np.ndarray:
        """
        Apply learned retargeting to skeleton pose.
        
        Args:
            q_skeleton: Skeleton joint angles
            
        Returns:
            q_anny: Predicted ANNY joint angles
        """
        # Start with neutral offset
        q_anny = self.params['neutral_offset'].copy()
        
        # Apply gains and copy relevant joints
        n_copy = min(len(q_skeleton), len(q_anny))
        for i in range(n_copy):
            if i < len(self.params['joint_gains']):
                q_anny[i] += self.params['joint_gains'][i] * q_skeleton[i]
        
        return q_anny
    
    def optimize(self, skeleton_poses: np.ndarray, 
                anny_poses: Optional[np.ndarray] = None,
                method: str = 'L-BFGS-B') -> Dict:
        """
        Optimize retargeting parameters.
        
        Args:
            skeleton_poses: Array of skeleton poses (N x D)
            anny_poses: Optional array of corresponding ANNY poses
            method: Optimization method ('L-BFGS-B' or 'differential_evolution')
            
        Returns:
            Optimization results
        """
        print(f"Starting optimization with {len(skeleton_poses)} poses...")
        
        # Initial parameters
        x0 = self._pack_params()
        
        # Define bounds
        bounds = []
        # Neutral offset bounds
        for _ in range(self.n_anny_joints):
            bounds.append((-np.pi, np.pi))
        # Joint gain bounds  
        for _ in range(len(self.params['joint_gains'])):
            bounds.append((0.1, 3.0))
        # Rotation bounds (axis-angle)
        for _ in range(3):
            bounds.append((-np.pi, np.pi))
        
        # Optimize
        if method == 'differential_evolution':
            result = differential_evolution(
                lambda x: self.objective_function(x, skeleton_poses, anny_poses),
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                disp=True
            )
        else:
            result = minimize(
                lambda x: self.objective_function(x, skeleton_poses, anny_poses),
                x0,
                method=method,
                bounds=bounds,
                options={'disp': True, 'maxiter': 100}
            )
        
        # Update parameters with optimized values
        self._unpack_params(result.x)
        
        print(f"Optimization complete. Final error: {result.fun:.6f}")
        
        return {
            'success': result.success,
            'error': result.fun,
            'params': self.params.copy(),
            'result': result
        }
    
    def save_parameters(self, filename: str):
        """Save optimized parameters to file."""
        np.savez(filename, **self.params)
        print(f"Parameters saved to {filename}")
    
    def load_parameters(self, filename: str):
        """Load parameters from file."""
        data = np.load(filename)
        self.params['neutral_offset'] = data['neutral_offset']
        self.params['joint_gains'] = data['joint_gains']
        self.params['rotation_offset'] = data['rotation_offset']
        print(f"Parameters loaded from {filename}")


class InteractiveCalibration:
    """
    Interactive calibration tool for manual pose matching.
    """
    
    def __init__(self, retargeter: OptimizedRetargeter):
        """
        Initialize interactive calibration.
        
        Args:
            retargeter: The OptimizedRetargeter instance
        """
        self.retargeter = retargeter
        self.calibration_poses = []
    
    def capture_pose_pair(self):
        """
        Capture current pose from both models for calibration.
        """
        q_skeleton = self.retargeter.skeleton_env._data.qpos.copy()
        q_anny = self.retargeter.anny_env._data.qpos.copy()
        
        self.calibration_poses.append({
            'skeleton': q_skeleton,
            'anny': q_anny
        })
        
        print(f"Captured pose pair #{len(self.calibration_poses)}")
    
    def calibrate_from_captures(self):
        """
        Run optimization using captured pose pairs.
        """
        if not self.calibration_poses:
            print("No poses captured yet!")
            return
        
        skeleton_poses = np.array([p['skeleton'] for p in self.calibration_poses])
        anny_poses = np.array([p['anny'] for p in self.calibration_poses])
        
        return self.retargeter.optimize(skeleton_poses, anny_poses)


def test_retargeting_with_squat():
    """
    Test the retargeting system with squat trajectory.
    """
    from loco_mujoco.environments.humanoids import SkeletonTorque
    from loco_mujoco.task_factories import ImitationFactory
    from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf
    
    print("Testing retargeting with squat trajectory...")
    
    # Create skeleton environment
    skeleton_env = SkeletonTorque()
    
    # Load squat trajectory
    dataset_conf = DefaultDatasetConf(task=["squat"], dataset_type="mocap")
    trajectory = ImitationFactory.get_default_traj(skeleton_env, dataset_conf)
    
    # Get trajectory joint positions
    skeleton_poses = np.array(trajectory.data.qpos)
    
    print(f"Loaded {len(skeleton_poses)} poses from squat trajectory")
    
    # Here you would:
    # 1. Create ANNY environment
    # 2. Create retargeter
    # 3. Either optimize from marker positions or use manual calibration
    # 4. Apply retargeting to full trajectory
    
    return skeleton_poses


if __name__ == "__main__":
    # Example usage
    print("Optimization-based retargeting system")
    
    # Test with squat trajectory
    skeleton_poses = test_retargeting_with_squat()
    
    print("\nTo use this system:")
    print("1. Create both skeleton and ANNY environments")
    print("2. Create OptimizedRetargeter instance")
    print("3. Either:")
    print("   a. Use optimize() with known pose correspondences")
    print("   b. Use InteractiveCalibration for manual matching")
    print("4. Apply retarget() to transform poses")
