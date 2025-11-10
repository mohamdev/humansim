#!/usr/bin/env python3
"""
Script to find body names in the MyoSkeleton model.
This helps identify the correct body names for specific anatomical parts.
"""

import mujoco
from loco_mujoco.environments.humanoids import MyoSkeleton


def main():
    """
    List all body names in the MyoSkeleton model.
    """
    print("Loading MyoSkeleton environment...")

    # Create the environment
    env = MyoSkeleton(disable_fingers=True)
    model = env._model

    print(f"\nTotal bodies in MyoSkeleton: {model.nbody}\n")
    print("=" * 80)
    print("ALL BODY NAMES")
    print("=" * 80)

    # Keywords to search for anatomical parts
    keywords = {
        'shoulder': [],
        'elbow': [],
        'head': [],
        'neck': [],
        'hip': [],
        'knee': [],
        'ankle': [],
        'wrist': [],
        'hand': [],
        'foot': [],
        'spine': [],
        'pelvis': [],
        'femur': [],
        'tibia': [],
        'humerus': [],
        'radius': [],
        'ulna': []
    }

    # Collect all body names
    all_bodies = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = f"body_{body_id}"

        all_bodies.append((body_id, body_name))

        # Check if body name contains any keyword
        body_name_lower = body_name.lower()
        for keyword in keywords:
            if keyword in body_name_lower:
                keywords[keyword].append((body_id, body_name))

    # Print bodies grouped by keyword
    print("\nBODIES GROUPED BY ANATOMICAL KEYWORDS:")
    print("=" * 80)

    for keyword, bodies in sorted(keywords.items()):
        if bodies:
            print(f"\n{keyword.upper()}:")
            for body_id, body_name in bodies:
                print(f"  [{body_id:3d}] {body_name}")

    # Print all bodies for reference
    print("\n\n" + "=" * 80)
    print("COMPLETE LIST OF ALL BODIES:")
    print("=" * 80)

    for body_id, body_name in all_bodies:
        print(f"[{body_id:3d}] {body_name}")

    print("\n" + "=" * 80)
    print(f"Total: {len(all_bodies)} bodies")
    print("=" * 80)


if __name__ == "__main__":
    main()
