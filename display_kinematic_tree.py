#!/usr/bin/env python3
"""
Script to display the kinematic tree structure of a MuJoCo model.
Shows all bodies and joints in a hierarchical tree format.
"""

import argparse
import mujoco
from loco_mujoco.task_factories import RLFactory


def print_kinematic_tree(model, body_id=0, indent=0, visited=None):
    """
    Recursively print the kinematic tree starting from a given body.

    Args:
        model: MuJoCo model
        body_id: Current body ID to print
        indent: Current indentation level
        visited: Set of visited body IDs to avoid cycles
    """
    if visited is None:
        visited = set()

    if body_id in visited:
        return
    visited.add(body_id)

    # Get body name
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if body_name is None:
        body_name = f"body_{body_id}"

    # Print the body
    prefix = "  " * indent
    print(f"{prefix}Body [{body_id}]: {body_name}")

    # Find and print joints attached to this body
    for jnt_id in range(model.njnt):
        if model.jnt_bodyid[jnt_id] == body_id:
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name is None:
                jnt_name = f"joint_{jnt_id}"

            # Get joint type
            jnt_type = model.jnt_type[jnt_id]
            type_names = {
                mujoco.mjtJoint.mjJNT_FREE: "FREE",
                mujoco.mjtJoint.mjJNT_BALL: "BALL",
                mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE",
                mujoco.mjtJoint.mjJNT_HINGE: "HINGE"
            }
            type_str = type_names.get(jnt_type, f"TYPE_{jnt_type}")

            print(f"{prefix}  └─ Joint [{jnt_id}]: {jnt_name} ({type_str})")

    # Recursively print child bodies
    for child_id in range(model.nbody):
        if model.body_parentid[child_id] == body_id and child_id != body_id:
            print_kinematic_tree(model, child_id, indent + 1, visited)


def print_tree_summary(model):
    """Print a summary of the model structure."""
    print("=" * 80)
    print("KINEMATIC TREE STRUCTURE")
    print("=" * 80)
    print(f"Total Bodies: {model.nbody}")
    print(f"Total Joints: {model.njnt}")
    print(f"Total DOFs: {model.nv}")
    print("=" * 80)
    print()


def print_all_bodies_and_joints(model):
    """Print a flat list of all bodies and joints."""
    print("\n" + "=" * 80)
    print("ALL BODIES (flat list)")
    print("=" * 80)
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = f"body_{body_id}"
        parent_id = model.body_parentid[body_id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        if parent_name is None:
            parent_name = f"body_{parent_id}"
        print(f"  [{body_id:3d}] {body_name:40s} (parent: [{parent_id:3d}] {parent_name})")

    print("\n" + "=" * 80)
    print("ALL JOINTS (flat list)")
    print("=" * 80)
    for jnt_id in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if jnt_name is None:
            jnt_name = f"joint_{jnt_id}"

        body_id = model.jnt_bodyid[jnt_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = f"body_{body_id}"

        # Get joint type
        jnt_type = model.jnt_type[jnt_id]
        type_names = {
            mujoco.mjtJoint.mjJNT_FREE: "FREE ",
            mujoco.mjtJoint.mjJNT_BALL: "BALL ",
            mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE",
            mujoco.mjtJoint.mjJNT_HINGE: "HINGE"
        }
        type_str = type_names.get(jnt_type, f"TYPE_{jnt_type}")

        # Get DOF range
        qpos_adr = model.jnt_qposadr[jnt_id]
        dof_adr = model.jnt_dofadr[jnt_id]

        print(f"  [{jnt_id:3d}] {jnt_name:40s} ({type_str}) on body [{body_id:3d}] {body_name} | qpos={qpos_adr}, dof={dof_adr}")


def main():
    """Main function to display kinematic tree."""
    parser = argparse.ArgumentParser(description="Display kinematic tree of a MuJoCo model")
    parser.add_argument(
        "--env",
        type=str,
        default="MyoSkeleton",
        help="Environment name (default: MyoSkeleton). Options: MyoSkeleton, Atlas, FourierGR1T2, ApptronikApollo"
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Display hierarchical tree structure"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Display flat list of bodies and joints"
    )

    args = parser.parse_args()

    # If no display option is specified, show both
    if not args.tree and not args.flat:
        args.tree = True
        args.flat = True

    print(f"Loading {args.env} environment...")

    try:
        env = RLFactory.make(args.env)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nAvailable environments: MyoSkeleton, Atlas, FourierGR1T2, ApptronikApollo")
        return

    # Get the MuJoCo model
    model = env._model

    print(f"Environment loaded successfully!\n")

    # Print summary
    print_tree_summary(model)

    # Print hierarchical tree
    if args.tree:
        print("HIERARCHICAL TREE (starting from world body)")
        print("=" * 80)
        print_kinematic_tree(model, body_id=0)

    # Print flat list
    if args.flat:
        print_all_bodies_and_joints(model)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
