#!/usr/bin/env python3
"""
Script to load and display the MyoSkeleton model with trajectory animation.
Locks unwanted joints (forearms and fingers) using joint damping and stiffness.

Bodies to keep:
- femurs, pelvis, abdomen, all lumbars, thoracic, all cervicals,
  scapulars, clavicles, humerus, patellas, tibias, talus, toes, calcn, lunate

All other bodies (fingers, radius, ulna, etc.) have their joints locked
and are hidden from visualization.
"""

import numpy as np
import mujoco
from loco_mujoco.environments.humanoids import MyoSkeleton
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf


# Joints to lock (forearms and all fingers)
JOINTS_TO_LOCK = [
    # Right arm
    'elbow_flex_r', 'pro_sup',
    # Right hand
    'cmc_flexion_r', 'cmc_abduction_r', 'mp_flexion_r', 'ip_flexion_r',
    'mcp2_flexion_r', 'mcp2_abduction_r', 'pm2_flexion_r', 'md2_flexion_r',
    'mcp3_flexion_r', 'mcp3_abduction_r', 'pm3_flexion_r', 'md3_flexion_r',
    'mcp4_flexion_r', 'mcp4_abduction_r', 'pm4_flexion_r', 'md4_flexion_r',
    'mcp5_flexion_r', 'mcp5_abduction_r', 'pm5_flexion_r', 'md5_flexion_r',
    # Left arm
    'elbow_flex_l', 'pro_sup_l',
    # Left hand
    'cmc_flexion_l', 'cmc_abduction_l', 'mp_flexion_l', 'ip_flexion_l',
    'mcp2_flexion_l', 'mcp2_abduction_l', 'pm2_flexion_l', 'md2_flexion_l',
    'mcp3_flexion_l', 'mcp3_abduction_l', 'pm3_flexion_l', 'md3_flexion_l',
    'mcp4_flexion_l', 'mcp4_abduction_l', 'pm4_flexion_l', 'md4_flexion_l',
    'mcp5_flexion_l', 'mcp5_abduction_l', 'pm5_flexion_l', 'md5_flexion_l',
]


# Define the bodies to KEEP visible
BODIES_TO_KEEP = [
    # Core and spine
    'pelvis',
    'abdomen',
    'lumbar1', 'lumbar2', 'lumbar3', 'lumbar4', 'lumbar5',
    'thoracic_spine',

    # Cervical spine
    'cerv1', 'cerv2', 'cerv3', 'cerv4', 'cerv5', 'cerv6', 'cerv7',

    # Head
    'skull',

    # Shoulder girdle
    'clavicle_r', 'clavicle_l',
    'scapula_r', 'scapula_l',
    'clavphant_r', 'clavphant_l',
    'scapphant_r', 'scapphant_l',

    # Upper arms
    'humphant_r', 'humphant1_r',
    'humphant_l', 'humphant1_l',
    'humerus_r', 'humerus_l',

    # Lower body
    'femur_r', 'femur_l',
    'patella_r', 'patella_l',
    'tibia_r', 'tibia_l',
    'talus_r', 'talus_l',
    'calcn_r', 'calcn_l',
    'toes_r', 'toes_l',

    # Hands (lunate only)
    'lunate_r', 'lunate_l',

    # Special bodies
    'world',
    'myoskeleton_root',
]


def lock_joints(model, data):
    """
    Lock unwanted joints by setting very high stiffness and damping.
    This effectively freezes them in their current position.

    Args:
        model: MuJoCo model
        data: MuJoCo data
    """
    print("\nLocking unwanted joints...")

    locked_count = 0
    locked_dofs = []

    for jnt_id in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)

        if jnt_name in JOINTS_TO_LOCK:
            # Get DOF address for this joint
            dof_adr = model.jnt_dofadr[jnt_id]

            # Set very high stiffness to lock the joint
            model.dof_armature[dof_adr] = 1000.0  # High inertia
            model.dof_damping[dof_adr] = 10000.0   # Very high damping

            # Set joint limits to current position (effectively locks it)
            qpos_adr = model.jnt_qposadr[jnt_id]
            current_pos = data.qpos[qpos_adr]

            # Limit the range to a very small window around current position
            model.jnt_range[jnt_id, 0] = current_pos - 0.001
            model.jnt_range[jnt_id, 1] = current_pos + 0.001

            locked_count += 1
            locked_dofs.append(dof_adr)

            if locked_count <= 5:  # Print first 5 for verification
                print(f"  Locked: {jnt_name} (DOF {dof_adr}, qpos={current_pos:.4f})")

    print(f"  Total joints locked: {locked_count}")
    print(f"  Total DOFs locked: {len(locked_dofs)}")

    return locked_dofs


def hide_unwanted_bodies(model):
    """
    Hide geoms of bodies that are not in the BODIES_TO_KEEP list.

    This is done by setting the geom group to a high number (group 5)
    and then disabling that group in the visualization options.

    Args:
        model: MuJoCo model
    """
    print("\nConfiguring body visibility...")

    hidden_count = 0
    visible_count = 0

    # Iterate through all geoms
    for geom_id in range(model.ngeom):
        # Get the body ID that this geom belongs to
        body_id = model.geom_bodyid[geom_id]

        # Get the body name
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = f"body_{body_id}"

        # Check if this body should be visible
        if body_name not in BODIES_TO_KEEP:
            # Move this geom to group 5 (which we'll disable)
            model.geom_group[geom_id] = 5
            hidden_count += 1
        else:
            # Keep in group 0 (default, visible)
            model.geom_group[geom_id] = 0
            visible_count += 1

    print(f"  Visible geoms: {visible_count}")
    print(f"  Hidden geoms: {hidden_count}")
    print(f"  Total geoms: {model.ngeom}")


def load_trajectory(env):
    """
    Load the walk motion trajectory from the default dataset.

    Args:
        env: The MyoSkeleton environment

    Returns:
        Trajectory object containing the motion data
    """
    print("\nLoading walk trajectory from default dataset...")

    # Create dataset configuration for walk motion
    dataset_conf = DefaultDatasetConf(task=["squat"], dataset_type="mocap")

    # Load the trajectory using ImitationFactory
    trajectory = ImitationFactory.get_default_traj(env, dataset_conf)

    print(f"Loaded trajectory with {trajectory.data.site_xpos.shape[0]} timesteps")
    print(f"Number of sites per timestep: {trajectory.data.site_xpos.shape[1]}")

    return trajectory


def main():
    """
    Load the MyoSkeleton environment and display animated trajectory
    with reduced body visualization.
    """
    print("Creating MyoSkeleton environment with reduced body visualization...")

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
    print(f"Total DOFs before locking: {env._model.nv}")

    # Reset environment to initial/neutral pose
    print("\nResetting to neutral pose...")
    env.reset()

    # Lock unwanted joints (must be done after reset to get proper qpos values)
    locked_dofs = lock_joints(env._model, env._data)

    # Hide unwanted bodies by modifying geom groups
    hide_unwanted_bodies(env._model)

    # Load the trajectory
    trajectory = load_trajectory(env)
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
        frames_configured = False

        while True:
            # Set the robot's joint positions from the trajectory
            env._data.qpos[:] = trajectory_qpos[traj_frame, :]

            # Compute forward kinematics to update all positions
            mujoco.mj_forward(env._model, env._data)

            # Configure visualization after the first render
            if not frames_configured and env._viewer is not None:
                # Enable coordinate frames for body segments
                env._viewer._scene_option.frame = mujoco.mjtFrame.mjFRAME_BODY

                # Enable body labels to see names
                env._viewer._scene_option.label = mujoco.mjtLabel.mjLABEL_BODY

                # Disable rendering of group 5 (where we put hidden geoms)
                env._viewer._scene_option.geomgroup[5] = 0

                frames_configured = True
                print("\n3D coordinate frames enabled!")
                print("Hidden bodies are not displayed.")
                print(f"\nNote: {len(locked_dofs)} DOFs are locked (fingers and forearms)")
                print("These joints still exist but are frozen in place via high damping.")
                print("For control, you can ignore these locked DOFs.")
                print("\nPlaying trajectory animation...")

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
