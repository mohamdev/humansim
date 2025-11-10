#!/usr/bin/env python3
"""
Script to load and display the SkeletonTorque model in neutral pose.

The SkeletonTorque is a humanoid skeleton model with torque actuators
(one per joint) that uses direct motor control. This script displays:
- 3D coordinate frames for each body segment
- Model in neutral/rest pose
"""

import mujoco
from loco_mujoco.environments.humanoids import SkeletonTorque


def main():
    """
    Load the SkeletonTorque environment and display in neutral pose.
    """
    print("Creating SkeletonTorque environment in neutral pose...")

    # Create the environment
    try:
        env = SkeletonTorque()
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nThe SkeletonTorque model should be automatically available after installation.")
        return

    print(f"\nEnvironment created successfully!")
    print(f"Action space dimension: {env.info.action_space.shape[0]}")
    print(f"Observation space dimension: {env.info.observation_space.shape[0]}")
    print(f"Number of bodies: {env._model.nbody}")

    # Reset environment to initial/neutral pose
    print("\nResetting to neutral pose...")
    env.reset()

    # Display the environment
    print("\nRendering environment...")
    print("Close the viewer window to exit.")
    print("\nViewer Controls:")
    print("  - Press E to toggle reference frames (coordinate axes)")
    print("  - Press T to make the model transparent")
    print("  - Press H to hide/show the menu")
    print("  - Press TAB to switch cameras")
    print("  - Drag with mouse to rotate view")
    print("  - Scroll to zoom")

    # Display the model in neutral pose
    try:
        frames_enabled = False

        while True:
            # Compute forward kinematics to update all positions
            mujoco.mj_forward(env._model, env._data)

            # Enable frame visualization after the first render
            if not frames_enabled and env._viewer is not None:
                # Enable coordinate frames for all body segments
                env._viewer._scene_option.frame = mujoco.mjtFrame.mjFRAME_BODY

                # Make frames twice smaller (default is 0.01 for width, 1.0 for length)
                env._model.vis.scale.framewidth *= 0.5
                env._model.vis.scale.framelength *= 0.5

                frames_enabled = True
                print("\n3D coordinate frames enabled!")
                print("Displaying neutral pose...")

            # Render the environment
            env.render()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        env.stop()


if __name__ == "__main__":
    main()
