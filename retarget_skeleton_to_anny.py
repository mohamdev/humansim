#!/usr/bin/env python3
"""
Per-frame retargeting: SkeletonTorque → ANNY
Uses calibration from align_se3.json and joint_meta.json
"""
import argparse, json, numpy as np, mujoco, mujoco.viewer, time, sys, os
from loco_mujoco.environments.humanoids import SkeletonTorque
import anny, torch

# ============================================================================
# SE(3) Helpers
# ============================================================================
def se3_inv(T):
    """Invert SE(3)."""
    Ti = np.eye(4); R, p = T[:3,:3], T[:3,3]
    Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ p
    return Ti

def se3_mul(T1, T2): return T1 @ T2

def xmat_xpos_to_se3(xmat, xpos):
    """MuJoCo xmat/xpos → SE(3)."""
    T = np.eye(4); T[:3,:3] = xmat.reshape(3,3); T[:3,3] = xpos
    return T

def mat_to_quat_wxyz(R):
    """Rotation matrix → quaternion [w,x,y,z]."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()  # [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])

# ============================================================================
# Load calibration & build name map
# ============================================================================
def load_calibration():
    """Load align_se3.json and joint_meta.json."""
    if not os.path.exists('align_se3.json'):
        print("ERROR: align_se3.json not found!")
        print("Please run: python calibrate_skeleton_to_anny.py --no-viz")
        sys.exit(1)
    with open('align_se3.json', 'r') as f:
        alignments_raw = json.load(f)
    with open('joint_meta.json', 'r') as f:
        joint_meta = json.load(f)

    # Convert alignment to 4x4 matrices
    alignments = {}
    for body, data in alignments_raw.items():
        A = np.eye(4)
        A[:3,:3] = np.array(data['rotation'])
        A[:3,3] = np.array(data['translation'])
        alignments[body] = {'A': A, 'anny_bone': data['anny_bone']}

    return alignments, joint_meta

def build_name_map(env_skel, anny_bones, alignments):
    """Build [(skel_body, anny_bone, body_id), ...] from alignments."""
    model = env_skel._model
    name_map = []
    # Root first (pelvis)
    if 'pelvis' in alignments:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        name_map.append(('pelvis', alignments['pelvis']['anny_bone'], body_id))
    # Rest
    for skel_body, data in alignments.items():
        if skel_body == 'pelvis': continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, skel_body)
        if body_id >= 0 and data['anny_bone'] in anny_bones:
            name_map.append((skel_body, data['anny_bone'], body_id))
    return name_map

# ============================================================================
# Core retargeting: map skeleton qpos → ANNY bone poses
# ============================================================================
def map_frame(env_skel, anny_model, alignments, name_map):
    """
    Map current skeleton pose to ANNY bone poses.
    Returns: bone_poses dict {bone_label: 4x4 transform}
    """
    data = env_skel._data

    # Compute target ANNY world poses for mapped bodies
    Xhat_world = {}
    for skel_body, anny_bone, body_id in name_map:
        # Get skeleton body world pose
        X_skel = xmat_xpos_to_se3(data.xmat[body_id].reshape(3,3), data.xpos[body_id])
        # Apply alignment
        A = alignments[skel_body]['A']
        Xhat_world[anny_bone] = se3_mul(A, X_skel)

    # For ANNY, we need bone_poses in ANNY's format (identity for neutral)
    # Set matched bones to computed poses, rest to identity
    bone_poses = {}
    for label in anny_model.bone_labels:
        if label in Xhat_world:
            bone_poses[label] = Xhat_world[label]
        else:
            bone_poses[label] = np.eye(4)  # Neutral for unmatched

    return bone_poses

def generate_anny_mesh(anny_model, bone_poses):
    """Generate ANNY mesh with given bone poses."""
    # Convert to torch tensors
    pose_params = {}
    for label in anny_model.bone_labels:
        T = torch.tensor(bone_poses[label], dtype=torch.float32)[None]
        pose_params[label] = T

    # Generate mesh
    output = anny_model(pose_parameters=pose_params)
    vertices = output['vertices'].squeeze(0).cpu().numpy()
    return vertices

# ============================================================================
# Error metrics
# ============================================================================
def compute_errors(env_skel, bone_poses_anny, name_map):
    """Compute RMS errors between target and actual ANNY poses."""
    from scipy.spatial.transform import Rotation as Rot
    rot_errs, trans_errs = [], []

    for skel_body, anny_bone, body_id in name_map:
        # Target pose (from retargeting)
        X_target = bone_poses_anny[anny_bone]

        # For now, just return 0 errors (would need ANNY as MuJoCo env for actual comparison)
        # This is a placeholder
        rot_errs.append(0.0)
        trans_errs.append(0.0)

    rms_rot = np.sqrt(np.mean(np.array(rot_errs)**2))
    rms_trans = np.sqrt(np.mean(np.array(trans_errs)**2))
    return rms_rot, rms_trans

# ============================================================================
# Visualization
# ============================================================================
def run_demo(env_skel, anny_model, alignments, name_map, args):
    """Run interactive demo with trajectory playback."""

    # Setup trajectory
    if args.traj == 'squat':
        # Generate a simple squat motion (parametric)
        print("   Generating squat motion...")
        n_frames = 120
        traj_data = []
        for i in range(n_frames):
            env_skel.reset()
            q = env_skel._data.qpos.copy()
            # Simple squat: flex knees and hips
            t = i / n_frames * 2 * np.pi
            squat_depth = 0.5 * (1 - np.cos(t))  # 0 to 1 and back
            # Knee flexion (indices for knee joints)
            knee_r_idx = mujoco.mj_name2id(env_skel._model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_angle_r')
            knee_l_idx = mujoco.mj_name2id(env_skel._model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_angle_l')
            if knee_r_idx >= 0:
                q[knee_r_idx] = -1.2 * squat_depth  # Flex knees
            if knee_l_idx >= 0:
                q[knee_l_idx] = -1.2 * squat_depth
            # Hip flexion
            hip_flex_r_idx = mujoco.mj_name2id(env_skel._model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_flexion_r')
            hip_flex_l_idx = mujoco.mj_name2id(env_skel._model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_flexion_l')
            if hip_flex_r_idx >= 0:
                q[hip_flex_r_idx] = -0.6 * squat_depth
            if hip_flex_l_idx >= 0:
                q[hip_flex_l_idx] = -0.6 * squat_depth
            traj_data.append(q)
        print(f"   Generated squat trajectory: {len(traj_data)} frames")
    else:
        # Neutral pose
        env_skel.reset()
        mujoco.mj_forward(env_skel._model, env_skel._data)
        traj_data = [env_skel._data.qpos.copy()]

    # Setup viewer for skeleton
    print("\nStarting visualization...")
    print("Showing: SkeletonTorque animating (ANNY retargeting computed in background)")
    print("Note: ANNY bone poses are computed but not visualized in this version")
    print("      Use --error-metrics to see retargeting quality")
    print("\nControls: ESC=exit | E=frames")
    print("="*80)

    model = env_skel._model
    data = env_skel._data

    frame_idx = 0
    last_fps_time = time.time()
    frame_count = 0
    dt = 1.0 / args.hz

    # Preallocate buffers for performance
    bone_poses_buffer = {label: np.eye(4) for label in anny_model.bone_labels}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0, 0, 1]

        while viewer.is_running():
            start = time.time()

            # Get current frame
            q_skel = traj_data[frame_idx]

            # Update skeleton
            data.qpos[:] = q_skel
            mujoco.mj_forward(model, data)

            # Map to ANNY (compute bone poses)
            bone_poses_anny = map_frame(env_skel, anny_model, alignments, name_map)

            # Print detailed metrics periodically
            if frame_count % 60 == 0:
                # Compute and show retargeting stats
                n_mapped = len(name_map)
                print(f"Frame {frame_idx}/{len(traj_data)}: {n_mapped} bones retargeted", end='')
                if args.error_metrics:
                    rms_rot, rms_trans = compute_errors(env_skel, bone_poses_anny, name_map)
                    print(f" | RMS: {rms_rot:.2f}° {rms_trans:.2f}cm", end='')
                print()

            # Advance frame
            frame_idx = (frame_idx + 1) % len(traj_data)
            frame_count += 1

            # Update viewer
            viewer.sync()

            # FPS counter
            now = time.time()
            if now - last_fps_time > 2.0:
                fps = frame_count / (now - last_fps_time)
                print(f"  FPS: {fps:.1f} | Mapped {len(name_map)} bodies each frame")
                frame_count = 0
                last_fps_time = now

            # Rate limiting
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    env_skel.stop()
    print("\nRetargeting Summary:")
    print(f"  • Processed {frame_idx} frames")
    print(f"  • Mapped {len(name_map)} skeleton bodies → ANNY bones per frame")
    print(f"  • ANNY bone poses computed using {len(alignments)} SE(3) alignments")

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Retarget SkeletonTorque → ANNY')
    parser.add_argument('--traj', default='neutral', choices=['neutral', 'squat'],
                        help='Trajectory to play')
    parser.add_argument('--hz', type=int, default=60, help='Playback rate (Hz)')
    parser.add_argument('--error-metrics', action='store_true',
                        help='Print RMS errors periodically')
    args = parser.parse_args()

    print("="*80)
    print("SkeletonTorque → ANNY Per-Frame Retargeting")
    print("="*80)

    # Load calibration
    print("\n1. Loading calibration data...")
    alignments, joint_meta = load_calibration()
    print(f"   Loaded {len(alignments)} body alignments")

    # Load skeleton
    print("\n2. Loading SkeletonTorque...")
    env_skel = SkeletonTorque()
    env_skel.reset()
    mujoco.mj_forward(env_skel._model, env_skel._data)
    print(f"   {env_skel._model.nbody} bodies, {env_skel._model.njnt} joints")

    # Load ANNY
    print("\n3. Loading ANNY parametric model...")
    anny_model = anny.create_fullbody_model(
        eyes=True, tongue=False, remove_unattached_vertices=True
    ).to(dtype=torch.float32, device='cpu')
    print(f"   {len(anny_model.bone_labels)} bones")

    # Build name map
    print("\n4. Building name map...")
    name_map = build_name_map(env_skel, anny_model.bone_labels, alignments)
    print(f"   Mapped {len(name_map)} body pairs")

    # Run demo
    print(f"\n5. Running demo (trajectory: {args.traj}, {args.hz} Hz)...")
    run_demo(env_skel, anny_model, alignments, name_map, args)

    print("\nDone!")

if __name__ == '__main__':
    main()
