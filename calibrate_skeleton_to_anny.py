#!/usr/bin/env python3
"""
One-time calibration: SkeletonTorque (ISB) → ANNY (parametric mesh).
Computes SE(3) body alignments and saves to JSON.

Usage: python calibrate_skeleton_to_anny.py [--no-viz] [--list-bodies]
"""
import argparse, json, numpy as np, mujoco, mujoco.viewer, tempfile, os
from typing import List, Tuple, Dict
from loco_mujoco.environments.humanoids import SkeletonTorque
import anny, torch

# ============================================================================
# MANUAL NAME MAP: Edit skeleton_body → anny_bone mappings here
# ============================================================================
MANUAL_NAME_MAP = {
    'pelvis': 'root', 'torso': 'spine03',
    'femur_r': 'upperleg01.R', 'femur_l': 'upperleg01.L',
    'tibia_r': 'lowerleg01.R', 'tibia_l': 'lowerleg01.L',
    'talus_r': 'foot.R', 'talus_l': 'foot.L',
    'humerus_r': 'shoulder01.R', 'humerus_l': 'shoulder01.L',
    'ulna_r': 'lowerarm01.R', 'ulna_l': 'lowerarm01.L',
    'hand_r': 'wrist.R', 'hand_l': 'wrist.L',
}

# ============================================================================
# SE(3) Utilities
# ============================================================================
def se3_inv(T):
    """Invert SE(3) matrix."""
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ p
    return Ti

def se3_mul(T1, T2): return T1 @ T2

def xmat_xpos_to_se3(xmat, xpos):
    """Convert MuJoCo xmat/xpos to SE(3)."""
    T = np.eye(4); T[:3, :3] = xmat.reshape(3, 3); T[:3, 3] = xpos
    return T

# ============================================================================
# Name mapping
# ============================================================================
def build_name_map(env_skel, anny_bones: List[str]) -> List[Tuple[str, str, int]]:
    """Build [(skel_body, anny_bone, body_id), ...] with root first."""
    model = env_skel._model
    name_map = []
    # Root first
    if 'pelvis' in MANUAL_NAME_MAP:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        name_map.append(('pelvis', MANUAL_NAME_MAP['pelvis'], pelvis_id))
    # Rest
    for skel_name, anny_name in MANUAL_NAME_MAP.items():
        if skel_name == 'pelvis': continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, skel_name)
        if body_id >= 0 and anny_name in anny_bones:
            name_map.append((skel_name, anny_name, body_id))
    return name_map

# ============================================================================
# Compute SE(3) alignments: A[bS] = X_anny @ inv(X_skel)
# ============================================================================
def compute_se3_alignments(env_skel, anny_bone_poses: Dict, name_map: List[Tuple]) -> Dict:
    """Compute alignment transforms in neutral pose."""
    data = env_skel._data
    alignments = {}
    for skel_body, anny_bone, body_id in name_map:
        X_skel = xmat_xpos_to_se3(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])
        X_anny = anny_bone_poses[anny_bone]
        A = se3_mul(X_anny, se3_inv(X_skel))
        alignments[skel_body] = {
            'rotation': A[:3, :3].tolist(),
            'translation': A[:3, 3].tolist(),
            'anny_bone': anny_bone
        }
    return alignments

# ============================================================================
# Visualization
# ============================================================================
def create_combined_scene(env_skel, anny_model, temp_dir, alignments):
    """Create combined MuJoCo XML with skeleton + ANNY mesh overlaid."""
    import trimesh, xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as Rot
    # Generate ANNY mesh
    pose_params = {label: torch.eye(4, dtype=torch.float32)[None] for label in anny_model.bone_labels}
    output = anny_model(pose_parameters=pose_params)
    vertices = output['vertices'].squeeze(0).cpu().numpy()
    faces = anny_model.faces.cpu().numpy()
    # Convert quads to triangles
    if faces.shape[1] == 4:
        tri = []
        for f in faces: tri.extend([[f[0], f[1], f[2]], [f[0], f[2], f[3]]])
        faces = np.array(tri)
    # Save mesh
    mesh_path = os.path.join(temp_dir, "anny_mesh.stl")
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)
    # Get skeleton XML path
    skel_xml = "./loco-mujoco/loco_mujoco/models/skeleton/skeleton_torque.xml"
    if not os.path.exists(skel_xml):
        # Fallback: find it in the loco_mujoco package
        import loco_mujoco
        pkg_dir = os.path.dirname(loco_mujoco.__file__)
        skel_xml = os.path.join(pkg_dir, "models/skeleton/skeleton_torque.xml")
    tree = ET.parse(skel_xml)
    root = tree.getroot()
    skel_meshdir = os.path.abspath(os.path.dirname(skel_xml))
    skel_assets = root.find('asset')
    skel_body = root.find('worldbody').find('body')
    # Build combined XML
    xml = f'''<?xml version="1.0"?>
<mujoco model="calibration">
    <compiler angle="radian" balanceinertia="true"/>
    <option timestep="0.01" gravity="0 0 -9.81"><flag contact="disable"/></option>
    <visual><scale framewidth="0.01" framelength="0.2"/></visual>
    <asset>
        <mesh name="anny_mesh" file="{mesh_path}"/>
'''
    # Add skeleton assets with absolute paths
    if skel_assets is not None:
        for c in skel_assets:
            asset_str = ET.tostring(c, encoding='unicode')
            # Convert relative mesh paths to absolute
            if 'file="' in asset_str and not asset_str.split('file="')[1].startswith('/'):
                rel_file = asset_str.split('file="')[1].split('"')[0]
                abs_file = os.path.join(skel_meshdir, rel_file)
                asset_str = asset_str.replace(f'file="{rel_file}"', f'file="{abs_file}"')
            xml += "        " + asset_str + "\n"
    xml += '''    </asset>
    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
        <body name="anny_root" pos="0 0 0">
            <freejoint name="anny_free"/>
            <geom name="anny_geom" type="mesh" mesh="anny_mesh" rgba="0.6 0.8 0.7 0.15"/>
        </body>
'''
    # Add skeleton body with root alignment applied
    import re
    # Get root (pelvis) alignment to position skeleton
    root_align = alignments.get('pelvis', None)
    if root_align:
        R_root = np.array(root_align['rotation'])
        t_root = np.array(root_align['translation'])
        # Convert rotation to quaternion (w,x,y,z)
        q = Rot.from_matrix(R_root).as_quat()  # [x,y,z,w]
        quat_wxyz = [q[3], q[0], q[1], q[2]]
        pos_str = f"{t_root[0]:.6f} {t_root[1]:.6f} {t_root[2]:.6f}"
        quat_str = f"{quat_wxyz[0]:.6f} {quat_wxyz[1]:.6f} {quat_wxyz[2]:.6f} {quat_wxyz[3]:.6f}"
        # Debug: print alignment being applied
        import sys
        if not hasattr(sys, '_alignment_printed'):
            print(f"   Root alignment: pos=[{t_root[0]:.3f}, {t_root[1]:.3f}, {t_root[2]:.3f}]")
            sys._alignment_printed = True
        # Create wrapper body with alignment transformation
        xml += f'        <body name="skel_aligned" pos="{pos_str}" quat="{quat_str}">\n'
    else:
        xml += '        <body name="skel_aligned" pos="0 0 0">\n'

    body_str = ET.tostring(skel_body, encoding='unicode')
    body_str = body_str.replace(' class="mimic"', '')
    body_str = re.sub(r'<freejoint[^/]*/>', '', body_str)
    xml += "            " + body_str + "\n"
    xml += "        </body>\n"
    xml += '''    </worldbody>
</mujoco>'''
    xml_path = os.path.join(temp_dir, "combined.xml")
    with open(xml_path, 'w') as f: f.write(xml)
    return xml_path

def visualize_alignment(env_skel, anny_model, bone_poses, alignments, name_map):
    """Show both models overlaid and compute RMS errors."""
    import shutil
    from scipy.spatial.transform import Rotation as Rot
    # Compute errors
    rot_errs, trans_errs = [], []
    for skel_body, anny_bone, body_id in name_map:
        X_skel = xmat_xpos_to_se3(env_skel._data.xmat[body_id].reshape(3, 3),
                                   env_skel._data.xpos[body_id])
        X_anny = bone_poses[anny_bone]
        A = np.eye(4)
        A[:3, :3] = np.array(alignments[skel_body]['rotation'])
        A[:3, 3] = np.array(alignments[skel_body]['translation'])
        X_pred = se3_mul(A, X_skel)
        R_err = X_pred[:3, :3].T @ X_anny[:3, :3]
        angle_err = np.linalg.norm(Rot.from_matrix(R_err).as_rotvec()) * 180 / np.pi
        trans_err = np.linalg.norm(X_pred[:3, 3] - X_anny[:3, 3]) * 100
        rot_errs.append(angle_err); trans_errs.append(trans_err)
    rms_rot = np.sqrt(np.mean(np.array(rot_errs)**2))
    rms_trans = np.sqrt(np.mean(np.array(trans_errs)**2))
    print(f"   RMS rotation error: {rms_rot:.2f} deg")
    print(f"   RMS translation error: {rms_trans:.2f} cm")
    # Visualize
    print("   Creating combined visualization...")
    print("   Applying root alignment to position skeleton over ANNY...")
    temp_dir = tempfile.mkdtemp()
    try:
        xml_path = create_combined_scene(env_skel, anny_model, temp_dir, alignments)
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        print("\n" + "="*80)
        print("Viewer Controls: E=frames | T=transparent | H=menu | ESC=exit")
        print("="*80)
        print("Showing: Skeleton (opaque) + ANNY (transparent)")
        print("         Skeleton has been aligned to overlay ANNY using root transformation")
        print("         Both models should be overlapping at the same position\n")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -10
            viewer.cam.lookat[:] = [0, 0, 1]
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
    finally:
        shutil.rmtree(temp_dir)

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Calibrate SkeletonTorque → ANNY')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--list-bodies', action='store_true', help='List body pairs and exit')
    parser.add_argument('--save', nargs=2, default=['align_se3.json', 'joint_meta.json'],
                        metavar=('ALIGN', 'JOINT'), help='Output JSON paths')
    args = parser.parse_args()

    print("="*80)
    print("SkeletonTorque → ANNY Calibration")
    print("="*80)

    # 1. Load SkeletonTorque
    print("\n1. Loading SkeletonTorque...")
    env_skel = SkeletonTorque()
    env_skel.reset()
    mujoco.mj_forward(env_skel._model, env_skel._data)
    print(f"   {env_skel._model.nbody} bodies, {env_skel._model.njnt} joints")

    # 2. Load ANNY
    print("\n2. Loading ANNY parametric model...")
    anny_model = anny.create_fullbody_model(eyes=True, tongue=False,
                                            remove_unattached_vertices=True
                                            ).to(dtype=torch.float32, device='cpu')
    pose_params = {label: torch.eye(4, dtype=torch.float32)[None] for label in anny_model.bone_labels}
    output = anny_model(pose_parameters=pose_params)
    bone_poses_tensor = output['bone_poses'].squeeze(0).cpu().numpy()
    bone_poses = {label: bone_poses_tensor[i] for i, label in enumerate(anny_model.bone_labels)}
    print(f"   {len(anny_model.bone_labels)} bones")

    # 3. Build name map
    print("\n3. Building name map...")
    name_map = build_name_map(env_skel, anny_model.bone_labels)
    print(f"   Mapped {len(name_map)} body pairs")

    if args.list_bodies:
        print("\nBody Pairs:")
        for skel_body, anny_bone, _ in name_map:
            print(f"  {skel_body:20s} → {anny_bone}")
        return

    # 4. Compute SE(3) alignments
    print("\n4. Computing SE(3) alignments...")
    alignments = compute_se3_alignments(env_skel, bone_poses, name_map)
    with open(args.save[0], 'w') as f:
        json.dump(alignments, f, indent=2)
    print(f"   Saved: {args.save[0]}")

    # 5. Compute joint metadata
    print("\n5. Computing joint metadata...")
    joint_meta = {}
    model = env_skel._model
    for i in range(model.njnt):
        jnt_type = model.jnt_type[i]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            joint_meta[jnt_name] = {
                'type': 'hinge', 'axis': model.jnt_axis[i].tolist(),
                'sign': 1.0, 'offset': 0.0
            }
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            joint_meta[jnt_name] = {'type': 'ball'}
        elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            joint_meta[jnt_name] = {'type': 'free'}
    with open(args.save[1], 'w') as f:
        json.dump(joint_meta, f, indent=2)
    print(f"   Saved: {args.save[1]}")

    # 6. Visual sanity check
    if not args.no_viz:
        print("\n6. Visual sanity check...")
        visualize_alignment(env_skel, anny_model, bone_poses, alignments, name_map)

    env_skel.stop()
    print("\nDone! Calibration complete.")

if __name__ == '__main__':
    main()
