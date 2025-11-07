#!/usr/bin/env python3
"""
Script to load and display the MyoSkeleton model in a neutral pose.

The MyoSkeleton is a biomechanical humanoid model with 151 joints
that accurately simulates human movement.
"""

import numpy as np
import mujoco
from loco_mujoco.task_factories import RLFactory


def main():
    """
    Load the MyoSkeleton environment and display it in a neutral pose.
    """
    print("Loading MyoSkeleton environment...")

    # Create the MyoSkeleton environment using RLFactory
    # Note: The MyoModel should be automatically available after installation
    try:
        env = RLFactory.make("MyoSkeleton")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nIf the model is missing, the loco-mujoco installation should have downloaded it.")
        print("You can also manually run 'loco-mujoco-myomodel-init' if needed.")
        return

    print(f"Environment created successfully!")
    print(f"Action space dimension: {env.info.action_space.shape[0]}")
    print(f"Observation space dimension: {env.info.observation_space.shape[0]}")

    # Reset environment to initial/neutral pose
    print("\nResetting to neutral pose...")
    env.reset()

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

    # Keep the environment in neutral pose and render
    # We apply zero actions to keep it still
    action_dim = env.info.action_space.shape[0]

    try:
        step = 0
        frames_enabled = False

        while True:
            # Apply zero action to maintain neutral pose
            # (you could also apply small random actions to see the model move)
            action = np.zeros(action_dim)

            # Step the environment
            observation, reward, absorbing, done, info = env.step(action)

            # Render the environment
            env.render()

            # Enable frame visualization after the first render
            # This shows the 3D coordinate frames for each body segment
            if not frames_enabled and env._viewer is not None:
                env._viewer._scene_option.frame = mujoco.mjtFrame.mjFRAME_BODY
                frames_enabled = True
                print("\n3D coordinate frames enabled for all body segments!")

            step += 1

            # Optional: reset every 1000 steps to maintain the pose
            if step % 1000 == 0:
                env.reset()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        env.stop()


if __name__ == "__main__":
    main()
