#!/usr/bin/env python3
"""
Analyze which joints need to be locked for unwanted bodies.
"""

import mujoco
from loco_mujoco.environments.humanoids import MyoSkeleton


# Bodies we want to REMOVE/LOCK (opposite of what we want to keep)
BODIES_TO_LOCK = [
    # Forearms (radius, ulna - we keep humerus)
    'radius_r', 'radius_l',
    'ulna_r', 'ulna_l',

    # Fingers (except lunate which we keep)
    'firstmc_r', 'firstmc_l',
    'proximal_thumb_r', 'proximal_thumb_l',
    'distal_thumb_r', 'distal_thumb_l',
    '2proxph_r', '2proxph_l',
    '2midph_r', '2midph_l',
    '2distph_r', '2distph_l',
    '3proxph_r', '3proxph_l',
    '3midph_r', '3midph_l',
    '3distph_r', '3distph_l',
    '4proxph_r', '4proxph_l',
    '4midph_r', '4midph_l',
    '4distph_r', '4distph_l',
    '5proxph_r', '5proxph_l',
    '5midph_r', '5midph_l',
    '5distph_r', '5distph_l',
]


def main():
    print("Loading MyoSkeleton environment...")
    env = MyoSkeleton(disable_fingers=True)
    model = env._model

    print(f"\nModel info:")
    print(f"  Total bodies: {model.nbody}")
    print(f"  Total joints: {model.njnt}")
    print(f"  Total DOFs (nv): {model.nv}")
    print(f"  Total qpos (nq): {model.nq}")

    print("\n" + "="*80)
    print("JOINTS ATTACHED TO BODIES WE WANT TO LOCK:")
    print("="*80)

    joints_to_lock = []

    for jnt_id in range(model.njnt):
        body_id = model.jnt_bodyid[jnt_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)

        if body_name in BODIES_TO_LOCK:
            # Get joint type
            jnt_type = model.jnt_type[jnt_id]
            type_names = {
                mujoco.mjtJoint.mjJNT_FREE: "FREE",
                mujoco.mjtJoint.mjJNT_BALL: "BALL",
                mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE",
                mujoco.mjtJoint.mjJNT_HINGE: "HINGE"
            }
            type_str = type_names.get(jnt_type, f"TYPE_{jnt_type}")

            qpos_adr = model.jnt_qposadr[jnt_id]
            dof_adr = model.jnt_dofadr[jnt_id]

            # Calculate DOF count for this joint
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                dof_count = 6
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                dof_count = 3
            else:
                dof_count = 1

            print(f"\nJoint [{jnt_id:3d}]: {jnt_name}")
            print(f"  Body: {body_name}")
            print(f"  Type: {type_str}")
            print(f"  qpos address: {qpos_adr}")
            print(f"  DOF address: {dof_adr}")
            print(f"  DOF count: {dof_count}")

            joints_to_lock.append({
                'id': jnt_id,
                'name': jnt_name,
                'body_name': body_name,
                'type': type_str,
                'dof_count': dof_count,
                'dof_adr': dof_adr
            })

    print("\n" + "="*80)
    print(f"SUMMARY:")
    print("="*80)
    print(f"Total joints to lock: {len(joints_to_lock)}")
    total_dofs_to_remove = sum(j['dof_count'] for j in joints_to_lock)
    print(f"Total DOFs to remove: {total_dofs_to_remove}")
    print(f"Remaining DOFs: {model.nv - total_dofs_to_remove}")


if __name__ == "__main__":
    main()
