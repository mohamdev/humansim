#!/usr/bin/env python3
"""
Script to load and display the MyoSkeleton model with trajectory animation.

The MyoSkeleton is a biomechanical humanoid model with 151 joints
that accurately simulates human movement. This script displays:
- 3D coordinate frames for each body segment
- Animated trajectory playback (squat motion)
"""

import numpy as np
import mujoco
from loco_mujoco.environments.humanoids import MyoSkeleton
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf


def load_squat_trajectory(env):
    """
    Load the squat motion trajectory from the default dataset.

    Args:
        env: The MyoSkeleton environment

    Returns:
        Trajectory object containing the squat motion data
    """
    print("\nLoading squat trajectory from default dataset...")

    # Create dataset configuration for squat motion
    dataset_conf = DefaultDatasetConf(task=["squat"], dataset_type="mocap")

    # Load the trajectory using ImitationFactory
    trajectory = ImitationFactory.get_default_traj(env, dataset_conf)

    print(f"Loaded trajectory with {trajectory.data.site_xpos.shape[0]} timesteps")
    print(f"Number of sites per timestep: {trajectory.data.site_xpos.shape[1]}")

    return trajectory


def main():
    """
    Load the MyoSkeleton environment and display animated trajectory.
    """
    print("Creating MyoSkeleton environment with trajectory animation...")

    # Create the environment
    try:
        env = MyoSkeleton(disable_fingers=True)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nThe MyoModel should be automatically available after installation.")
        return

    print(f"\nEnvironment created successfully!")
    print(f"Action space dimension: {env.info.action_space.shape[0]}")
    print(f"Observation space dimension: {env.info.observation_space.shape[0]}")
    print(f"Number of bodies: {env._model.nbody}")

    # Reset environment to initial/neutral pose
    print("\nResetting to neutral pose...")
    env.reset()

    # Load the squat trajectory
    trajectory = load_squat_trajectory(env)
    num_timesteps = trajectory.data.qpos.shape[0]

    # Extract joint positions from trajectory
    trajectory_qpos = np.array(trajectory.data.qpos)

    print(f"\nTrajectory loaded:")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Joints (qpos dim): {trajectory_qpos.shape[1]}")

    # Display the environment
    print("Rendering environment...")
    print("Close the viewer window to exit.")
    print("\nViewer Controls:")
    print("  - Press SPACE to pause/unpause")
    print("  - Press E to toggle reference frames (coordinate axes)")
    print("  - Press T to make the model transparent")
    print("  - Press H to hide/show the menu")
    print("  - Press TAB to switch cameras")
    print("  - Drag with mouse to rotate view")
    print("  - Scroll to zoom")

    # Play the trajectory by setting qpos from the trajectory data
    try:
        step = 0
        traj_frame = 0
        frames_enabled = False

        while True:
            # Set the robot's joint positions from the trajectory
            # This positions the robot in the correct pose for this frame
            env._data.qpos[:] = trajectory_qpos[traj_frame, :]

            # Compute forward kinematics to update all positions
            mujoco.mj_forward(env._model, env._data)

            # Enable frame visualization after the first render
            if not frames_enabled and env._viewer is not None:
                # Enable coordinate frames for all body segments
                env._viewer._scene_option.frame = mujoco.mjtFrame.mjFRAME_BODY

                frames_enabled = True
                print("\n3D coordinate frames enabled!")
                print("Playing trajectory animation...")

            # Render the environment
            env.render()

            step += 1
            traj_frame = (traj_frame + 1) % num_timesteps  # Loop through trajectory

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        env.stop()


if __name__ == "__main__":
    main()