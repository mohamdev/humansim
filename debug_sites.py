#!/usr/bin/env python3
"""
Debug script to inspect site names and IDs in the MyoSkeleton model and trajectory.
"""

import numpy as np
import mujoco
from loco_mujoco.environments.humanoids import MyoSkeleton
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf


def main():
    print("Creating MyoSkeleton environment...")

    # Create environment
    env = MyoSkeleton(disable_fingers=True)
    env.reset()

    # Print all site names and IDs in the model
    print("\n" + "="*80)
    print("SITES IN THE MODEL:")
    print("="*80)
    print(f"Total number of sites: {env._model.nsite}")
    print("\nSite ID -> Site Name mapping:")
    for i in range(env._model.nsite):
        site_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_SITE, i)
        site_body_id = env._data.site_xpos[i]
        print(f"  {i:3d}: {site_name}")

    # Load the squat trajectory
    print("\n" + "="*80)
    print("LOADING SQUAT TRAJECTORY:")
    print("="*80)
    dataset_conf = DefaultDatasetConf(task=["squat"], dataset_type="mocap")
    trajectory = ImitationFactory.get_default_traj(env, dataset_conf)

    print(f"Trajectory shape: {trajectory.data.site_xpos.shape}")
    print(f"  Timesteps: {trajectory.data.site_xpos.shape[0]}")
    print(f"  Sites: {trajectory.data.site_xpos.shape[1]}")
    print(f"  Coordinates: {trajectory.data.site_xpos.shape[2]}")

    # Check if trajectory has site names
    if hasattr(trajectory, 'site_names'):
        print("\nTrajectory site names:")
        for i, name in enumerate(trajectory.site_names):
            print(f"  {i:3d}: {name}")
    else:
        print("\nTrajectory does not have site_names attribute")

    # Check trajectory data attributes
    print("\n" + "="*80)
    print("TRAJECTORY DATA ATTRIBUTES:")
    print("="*80)
    for attr in dir(trajectory.data):
        if not attr.startswith('_'):
            value = getattr(trajectory.data, attr)
            if isinstance(value, np.ndarray):
                print(f"  {attr}: shape = {value.shape}")
            else:
                print(f"  {attr}: {type(value).__name__}")

    # Sample first frame positions to see which sites are moving
    print("\n" + "="*80)
    print("FIRST FRAME SITE POSITIONS (first 25 sites):")
    print("="*80)
    first_frame = trajectory.data.site_xpos[0]
    for i in range(min(25, first_frame.shape[0])):
        pos = first_frame[i]
        # Get site name from model
        if i < env._model.nsite:
            site_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_SITE, i)
            print(f"  {i:3d} ({site_name}): x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")
        else:
            print(f"  {i:3d}: x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")

    # Check for sites at origin (0,0,0) or near origin
    print("\n" + "="*80)
    print("SITES NEAR ORIGIN (< 0.1 from origin) in first frame:")
    print("="*80)
    for i in range(min(first_frame.shape[0], env._model.nsite)):
        pos = first_frame[i]
        dist = np.linalg.norm(pos)
        if dist < 0.1:
            site_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_SITE, i)
            print(f"  {i:3d} ({site_name}): distance from origin = {dist:.6f}")

    env.stop()


if __name__ == "__main__":
    main()
