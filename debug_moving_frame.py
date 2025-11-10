#!/usr/bin/env python3
"""
Debug script to identify a frame that moves during animation.
This script tracks body positions between the first and subsequent frames
to identify which body's coordinate frame is moving unexpectedly.
"""

import numpy as np
import mujoco
from loco_mujoco.environments.humanoids import MyoSkeleton
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf


def main():
    """
    Track body positions to identify the moving frame.
    """
    print("Creating MyoSkeleton environment...")

    # Create the environment
    env = MyoSkeleton(disable_fingers=True)

    # Reset environment to initial/neutral pose
    env.reset()

    # Load the squat trajectory
    print("\nLoading squat trajectory...")
    dataset_conf = DefaultDatasetConf(task=["squat"], dataset_type="mocap")
    trajectory = ImitationFactory.get_default_traj(env, dataset_conf)
    trajectory_qpos = np.array(trajectory.data.qpos)

    # Store initial body positions (frame 0)
    env._data.qpos[:] = trajectory_qpos[0, :]
    mujoco.mj_forward(env._model, env._data)

    initial_body_positions = {}
    for body_id in range(env._model.nbody):
        body_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = f"body_{body_id}"
        initial_body_positions[body_id] = {
            'name': body_name,
            'pos': env._data.xpos[body_id].copy()
        }

    print(f"\nTracking {len(initial_body_positions)} bodies across trajectory...")
    print("Looking for bodies that move significantly in early frames...\n")

    # Check positions at frame 10 and frame 50
    check_frames = [1, 5, 10, 20, 50]

    for check_frame in check_frames:
        if check_frame >= trajectory_qpos.shape[0]:
            continue

        print(f"\n{'='*80}")
        print(f"Frame {check_frame} - Bodies with large position changes:")
        print(f"{'='*80}")

        env._data.qpos[:] = trajectory_qpos[check_frame, :]
        mujoco.mj_forward(env._model, env._data)

        movements = []
        for body_id in range(env._model.nbody):
            current_pos = env._data.xpos[body_id]
            initial_pos = initial_body_positions[body_id]['pos']

            # Calculate displacement
            displacement = np.linalg.norm(current_pos - initial_pos)

            # Store for sorting
            movements.append({
                'id': body_id,
                'name': initial_body_positions[body_id]['name'],
                'displacement': displacement,
                'initial_pos': initial_pos,
                'current_pos': current_pos,
                'delta': current_pos - initial_pos
            })

        # Sort by displacement (largest first)
        movements.sort(key=lambda x: x['displacement'], reverse=True)

        # Print top 10 movers
        for i, mov in enumerate(movements[:10]):
            print(f"{i+1:2d}. Body [{mov['id']:3d}] {mov['name']:30s}")
            print(f"    Displacement: {mov['displacement']:.4f} m")
            print(f"    Initial:  [{mov['initial_pos'][0]:7.4f}, {mov['initial_pos'][1]:7.4f}, {mov['initial_pos'][2]:7.4f}]")
            print(f"    Current:  [{mov['current_pos'][0]:7.4f}, {mov['current_pos'][1]:7.4f}, {mov['current_pos'][2]:7.4f}]")
            print(f"    Delta:    [{mov['delta'][0]:7.4f}, {mov['delta'][1]:7.4f}, {mov['delta'][2]:7.4f}]")
            print()

    # Now check for bodies that are far from the main body cluster at frame 0
    print(f"\n{'='*80}")
    print("Bodies far from origin at frame 0 (potential outliers):")
    print(f"{'='*80}")

    env._data.qpos[:] = trajectory_qpos[0, :]
    mujoco.mj_forward(env._model, env._data)

    outliers = []
    for body_id in range(env._model.nbody):
        body_name = initial_body_positions[body_id]['name']
        pos = env._data.xpos[body_id]
        distance_from_origin = np.linalg.norm(pos)

        outliers.append({
            'id': body_id,
            'name': body_name,
            'pos': pos,
            'distance': distance_from_origin
        })

    # Sort by distance from origin
    outliers.sort(key=lambda x: x['distance'], reverse=True)

    # Print bodies that are unusually far
    print("\nTop 10 bodies by distance from origin:")
    for i, outl in enumerate(outliers[:10]):
        print(f"{i+1:2d}. Body [{outl['id']:3d}] {outl['name']:30s}")
        print(f"    Position: [{outl['pos'][0]:7.4f}, {outl['pos'][1]:7.4f}, {outl['pos'][2]:7.4f}]")
        print(f"    Distance from origin: {outl['distance']:.4f} m")
        print()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
