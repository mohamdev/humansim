#!/usr/bin/env python3
"""
Script to inspect the contents and dimensions of the trajectory dataset.
"""

import numpy as np
import mujoco
from loco_mujoco.environments.humanoids import MyoSkeleton
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf


def main():
    print("="*80)
    print("INSPECTING TRAJECTORY DATASET CONTENTS")
    print("="*80)

    # Create environment
    print("\nCreating MyoSkeleton environment...")
    env = MyoSkeleton(disable_fingers=True)
    env.reset()

    # Print model dimensions
    print("\n" + "="*80)
    print("MODEL DIMENSIONS:")
    print("="*80)
    print(f"Number of joints (nq): {env._model.nq}")
    print(f"Number of velocity DoFs (nv): {env._model.nv}")
    print(f"Number of bodies: {env._model.nbody}")
    print(f"Number of sites: {env._model.nsite}")
    print(f"Number of actuators: {env._model.nu}")

    # Load the squat trajectory
    print("\n" + "="*80)
    print("LOADING SQUAT TRAJECTORY:")
    print("="*80)
    dataset_conf = DefaultDatasetConf(task=["squat"], dataset_type="mocap")
    trajectory = ImitationFactory.get_default_traj(env, dataset_conf)

    print(f"\nTrajectory type: {type(trajectory)}")
    print(f"Number of timesteps: {trajectory.data.site_xpos.shape[0]}")

    # Inspect all data arrays in the trajectory
    print("\n" + "="*80)
    print("TRAJECTORY DATA CONTENTS:")
    print("="*80)

    # Check qpos
    if hasattr(trajectory.data, 'qpos'):
        qpos = trajectory.data.qpos
        print(f"\n1. JOINT POSITIONS (qpos):")
        print(f"   Shape: {qpos.shape}")
        print(f"   Type: {type(qpos)}")
        print(f"   Description: Joint positions (generalized coordinates)")
        print(f"   Dimensions: (timesteps={qpos.shape[0]}, nq={qpos.shape[1]})")
        print(f"   - nq includes: free joint (7) + body joints")
        print(f"   - Free joint: 3D position (3) + quaternion (4) = 7 DoFs")
        print(f"\n   Sample from first timestep (first 20 values):")
        print(f"   {qpos[0, :20]}")

    # Check qvel
    if hasattr(trajectory.data, 'qvel'):
        qvel = trajectory.data.qvel
        print(f"\n2. JOINT VELOCITIES (qvel):")
        print(f"   Shape: {qvel.shape}")
        print(f"   Type: {type(qvel)}")
        print(f"   Description: Joint velocities")
        print(f"   Dimensions: (timesteps={qvel.shape[0]}, nv={qvel.shape[1]})")

    # Check site_xpos
    if hasattr(trajectory.data, 'site_xpos'):
        site_xpos = trajectory.data.site_xpos
        print(f"\n3. SITE POSITIONS (site_xpos):")
        print(f"   Shape: {site_xpos.shape}")
        print(f"   Type: {type(site_xpos)}")
        print(f"   Description: 3D Cartesian positions of sites in world frame")
        print(f"   Dimensions: (timesteps={site_xpos.shape[0]}, num_sites={site_xpos.shape[1]}, xyz={site_xpos.shape[2]})")
        print(f"\n   Sites in the model:")
        for i in range(env._model.nsite):
            site_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_SITE, i)
            pos = site_xpos[0, i, :]
            print(f"   - Site {i:2d} ({site_name:20s}): x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")

    # Check site_xmat
    if hasattr(trajectory.data, 'site_xmat'):
        site_xmat = trajectory.data.site_xmat
        print(f"\n4. SITE ORIENTATIONS (site_xmat):")
        print(f"   Shape: {site_xmat.shape}")
        print(f"   Type: {type(site_xmat)}")
        print(f"   Description: 3x3 rotation matrices for site orientations")
        print(f"   Dimensions: (timesteps={site_xmat.shape[0]}, num_sites={site_xmat.shape[1]}, 3x3_matrix={site_xmat.shape[2]})")

    # Check body positions
    if hasattr(trajectory.data, 'xpos'):
        xpos = trajectory.data.xpos
        print(f"\n5. BODY POSITIONS (xpos):")
        print(f"   Shape: {xpos.shape}")
        print(f"   Type: {type(xpos)}")
        print(f"   Description: 3D Cartesian positions of bodies in world frame")
        print(f"   Dimensions: (timesteps={xpos.shape[0]}, num_bodies={xpos.shape[1]}, xyz={xpos.shape[2]})")

    # Check body orientations
    if hasattr(trajectory.data, 'xquat'):
        xquat = trajectory.data.xquat
        print(f"\n6. BODY ORIENTATIONS (xquat):")
        print(f"   Shape: {xquat.shape}")
        print(f"   Type: {type(xquat)}")
        print(f"   Description: Quaternions for body orientations")
        print(f"   Dimensions: (timesteps={xquat.shape[0]}, num_bodies={xquat.shape[1]}, quaternion={xquat.shape[2]})")

    # Check center of mass velocities
    if hasattr(trajectory.data, 'cvel'):
        cvel = trajectory.data.cvel
        print(f"\n7. BODY CENTER OF MASS VELOCITIES (cvel):")
        print(f"   Shape: {cvel.shape}")
        print(f"   Type: {type(cvel)}")
        print(f"   Description: 6D velocities (linear + angular) of body CoMs")
        print(f"   Dimensions: (timesteps={cvel.shape[0]}, num_bodies={cvel.shape[1]}, vel_6d={cvel.shape[2]})")

    # Check subtree COM
    if hasattr(trajectory.data, 'subtree_com'):
        subtree_com = trajectory.data.subtree_com
        print(f"\n8. SUBTREE CENTER OF MASS (subtree_com):")
        print(f"   Shape: {subtree_com.shape}")
        print(f"   Type: {type(subtree_com)}")
        print(f"   Description: Center of mass of body subtrees")
        print(f"   Dimensions: (timesteps={subtree_com.shape[0]}, num_bodies={subtree_com.shape[1]}, xyz={subtree_com.shape[2]})")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("""
The trajectory dataset contains:

1. **Joint Configurations (qpos)**:
   - Generalized coordinates for all joints
   - Includes root position (3D) and orientation (quaternion)
   - This is the minimal representation - from this, all other positions can be computed

2. **Kinematics Data (site_xpos, xpos, xquat)**:
   - Pre-computed 3D positions and orientations
   - Derived from qpos via forward kinematics
   - Useful for visualization and reward computation

3. **Velocity Data (qvel, cvel)**:
   - Joint velocities and body velocities
   - Can be used for dynamics and control

To replay the motion, you only need to set qpos at each timestep and call mj_forward().
The sites will automatically be positioned correctly based on the joint configuration.
""")

    env.stop()


if __name__ == "__main__":
    main()
