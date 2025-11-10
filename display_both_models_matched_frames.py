#!/usr/bin/env python3
"""
Script to display both ANNY and SkeletonTorque models with matched anatomical frames.

This script identifies common anatomical landmarks (shoulders, elbows, wrists, lumbar,
hips, knees, ankles) in both models and displays coordinate frames only at these
matched locations for easy comparison.

Features:
- Matched anatomical frames at 14 common landmarks
- More transparent ANNY mesh (20% opacity)
- Larger coordinate frames for better visibility
- World origin frame (extra large) at (0, 0, 0)
"""

import torch
import numpy as np
import mujoco
import mujoco.viewer
import anny
import tempfile
import os
import shutil
from loco_mujoco.environments.humanoids import SkeletonTorque


# Define anatomical landmark mappings between ANNY and SkeletonTorque
ANATOMICAL_MATCHES = {
    # Format: 'landmark_name': ('anny_bone_label', 'skeleton_body_name')
    'left_shoulder': ('shoulder01.L', 'humerus_l'),
    'right_shoulder': ('shoulder01.R', 'humerus_r'),
    'left_elbow': ('lowerarm01.L', 'ulna_l'),
    'right_elbow': ('lowerarm01.R', 'ulna_r'),
    'left_wrist': ('wrist.L', 'hand_l'),
    'right_wrist': ('wrist.R', 'hand_r'),
    'left_hip': ('upperleg01.L', 'femur_l'),
    'right_hip': ('upperleg01.R', 'femur_r'),
    'left_knee': ('lowerleg01.L', 'tibia_l'),
    'right_knee': ('lowerleg01.R', 'tibia_r'),
    'left_ankle': ('foot.L', 'talus_l'),
    'right_ankle': ('foot.R', 'talus_r'),
    'lumbar': ('spine03', 'torso'),
    'pelvis': ('root', 'pelvis'),
}


def create_anny_mesh_and_matched_bones(anny_model, matched_bones):
    """
    Generate ANNY mesh in neutral pose and extract matched bone poses.

    Args:
        anny_model: The ANNY model instance
        matched_bones: List of ANNY bone labels to extract

    Returns:
        vertices: numpy array of mesh vertices
        faces: numpy array of mesh faces
        matched_bone_poses: dict mapping bone label to 4x4 transformation matrix
    """
    print("Generating ANNY mesh in neutral pose...")

    batch_size = 1

    # Set all bone transformations to identity (neutral/rest pose)
    pose_parameters = {
        label: torch.eye(4)[None].repeat(batch_size, 1, 1)
        for label in anny_model.bone_labels
    }

    # Generate the mesh
    output = anny_model(pose_parameters=pose_parameters)

    # Extract vertices and convert to numpy
    vertices = output['vertices'].squeeze(0).cpu().numpy()
    faces = anny_model.faces.cpu().numpy()

    # Extract bone poses for matched bones only
    all_bone_poses = output['bone_poses'].squeeze(0).cpu().numpy()
    matched_bone_poses = {}

    for bone_label in matched_bones:
        if bone_label in anny_model.bone_labels:
            bone_idx = anny_model.bone_labels.index(bone_label)
            matched_bone_poses[bone_label] = all_bone_poses[bone_idx]

    print(f"Generated ANNY mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"Extracted {len(matched_bone_poses)} matched bone poses")

    return vertices, faces, matched_bone_poses


def get_skeleton_body_positions(skeleton_env, matched_bodies):
    """
    Extract body positions from SkeletonTorque model.

    Args:
        skeleton_env: SkeletonTorque environment
        matched_bodies: List of body names to extract

    Returns:
        matched_body_positions: dict mapping body name to 3D position
    """
    print("Extracting SkeletonTorque body positions...")

    # Reset to neutral pose
    skeleton_env.reset()
    mujoco.mj_forward(skeleton_env._model, skeleton_env._data)

    matched_body_positions = {}

    for body_name in matched_bodies:
        # Find body ID
        body_id = mujoco.mj_name2id(skeleton_env._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            # Get body position from xpos (body position in world coordinates)
            body_pos = skeleton_env._data.xpos[body_id].copy()
            matched_body_positions[body_name] = body_pos

    print(f"Extracted {len(matched_body_positions)} skeleton body positions")

    return matched_body_positions


def write_stl(filename, vertices, faces):
    """Write a simple binary STL file."""
    with open(filename, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(np.uint32(len(faces)).tobytes())

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-8)

            f.write(normal.astype(np.float32).tobytes())
            f.write(v0.astype(np.float32).tobytes())
            f.write(v1.astype(np.float32).tobytes())
            f.write(v2.astype(np.float32).tobytes())
            f.write(np.uint16(0).tobytes())


def get_skeleton_xml_path():
    """Get the path to the SkeletonTorque XML file."""
    local_path = "./loco-mujoco/loco_mujoco/models/skeleton/skeleton_torque.xml"
    if os.path.exists(local_path):
        return os.path.abspath(local_path)

    env = SkeletonTorque()
    xml_path = env._model_path
    env.stop()
    return xml_path


def create_combined_xml_with_matched_frames(
    anny_vertices, anny_faces, anny_matched_poses, anny_bone_labels,
    skeleton_matched_positions, skeleton_body_names,
    anatomical_matches, temp_dir
):
    """
    Create combined MuJoCo XML with only matched anatomical frames displayed.

    Args:
        anny_vertices: ANNY mesh vertices
        anny_faces: ANNY mesh faces
        anny_matched_poses: Dict of ANNY bone poses
        anny_bone_labels: List of matched ANNY bone labels
        skeleton_matched_positions: Dict of skeleton body positions
        skeleton_body_names: List of matched skeleton body names
        anatomical_matches: Dict of anatomical landmark mappings
        temp_dir: Temporary directory for files

    Returns:
        xml_path: Path to combined XML file
    """
    mesh_path = os.path.join(temp_dir, "anny_mesh.stl")
    xml_path = os.path.join(temp_dir, "combined_matched.xml")

    # Convert ANNY faces to triangles if needed
    if anny_faces.shape[1] == 4:
        print("Converting ANNY quad faces to triangles...")
        triangles = []
        for face in anny_faces:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        faces_tri = np.array(triangles)
    else:
        faces_tri = anny_faces

    # Save ANNY mesh
    import trimesh
    mesh = trimesh.Trimesh(vertices=anny_vertices, faces=faces_tri, process=False)
    mesh.export(mesh_path)
    print(f"Saved ANNY mesh to {mesh_path}")

    # Generate ANNY matched bone sites (small size and low alpha to minimize sphere visibility)
    anny_sites_xml = ""
    for landmark_name, (anny_bone, _) in anatomical_matches.items():
        if anny_bone in anny_matched_poses:
            pose = anny_matched_poses[anny_bone]
            pos = pose[:3, 3]
            safe_name = landmark_name.replace(' ', '_')
            anny_sites_xml += f'                <site name="anny_{safe_name}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" size="0.005" rgba="1 0 0 0.2"/>\n'

    # Load SkeletonTorque XML
    import xml.etree.ElementTree as ET
    original_skeleton_xml_path = get_skeleton_xml_path()
    skeleton_tree = ET.parse(original_skeleton_xml_path)
    skeleton_root = skeleton_tree.getroot()

    # Extract skeleton components
    skeleton_worldbody = skeleton_root.find('worldbody')
    skeleton_body = skeleton_worldbody.find('body')

    # Convert skeleton body to string with prefixes
    skeleton_body_str = ET.tostring(skeleton_body, encoding='unicode')
    skeleton_body_str = skeleton_body_str.replace('name="', 'name="skel_')
    skeleton_body_str = skeleton_body_str.replace('joint="', 'joint="skel_')
    skeleton_body_str = skeleton_body_str.replace('mesh="', 'mesh="skel_')
    skeleton_body_str = skeleton_body_str.replace(' class="mimic"', '')

    # Remove freejoint
    import re
    skeleton_body_str = re.sub(r'<freejoint[^/]*/>',  '', skeleton_body_str)
    skeleton_body_str = re.sub(r'<freejoint[^>]*>[^<]*</freejoint>', '', skeleton_body_str)

    # Get skeleton directories
    skeleton_meshdir = os.path.dirname(get_skeleton_xml_path())

    # Extract defaults
    skeleton_defaults = skeleton_root.find('default')
    skeleton_defaults_str = ""
    if skeleton_defaults is not None:
        for child in skeleton_defaults:
            default_str = ET.tostring(child, encoding='unicode')
            skeleton_defaults_str += "            " + default_str + "\n"

    # Extract assets
    skeleton_assets = skeleton_root.find('asset')
    skeleton_assets_str = ""
    if skeleton_assets is not None:
        for child in skeleton_assets:
            asset_str = ET.tostring(child, encoding='unicode')
            asset_str = asset_str.replace('name="', 'name="skel_')
            asset_str = asset_str.replace('mesh="', 'mesh="skel_')
            if 'file="' in asset_str and not asset_str.split('file="')[1].startswith('/'):
                asset_str = asset_str.replace('file="', f'file="{skeleton_meshdir}/')
            skeleton_assets_str += "            " + asset_str + "\n"

    # Generate skeleton matched body sites
    skeleton_sites_xml = ""
    for landmark_name, (_, skeleton_body) in anatomical_matches.items():
        if skeleton_body in skeleton_matched_positions:
            # We'll add sites relative to the skeleton offset body
            # Skeleton is at (0.5, 0, 0), so we need to adjust positions
            pass  # Sites will be added via the body structure

    # Create combined XML
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="matched_frames">
        <compiler angle="radian" meshdir="{temp_dir}" balanceinertia="true" inertiafromgeom="auto"/>

        <option timestep="0.01" gravity="0 0 -9.81">
            <flag contact="disable"/>
        </option>

        <default>
{skeleton_defaults_str}
        </default>

        <visual>
            <global offwidth="1920" offheight="1080"/>
            <quality shadowsize="4096"/>
            <map force="0.1" fogstart="5" fogend="10"/>
            <scale framewidth="0.015" framelength="0.3"/>
        </visual>

        <asset>
            <mesh name="anny_mesh" file="anny_mesh.stl" scale="1 1 1"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".3 .4 .5" width="512" height="512"/>
            <texture name="texgeom" type="2d" builtin="flat" mark="cross" width="128" height="128" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            <material name="anny_material" rgba="0.6 0.8 0.7 0.1"/>

{skeleton_assets_str}
        </asset>

        <worldbody>
            <light directional="true" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5" dir="0 0 -1"/>
            <light directional="true" diffuse=".4 .4 .4" pos="0 0 3" dir="0 1 -1"/>

            <geom name="floor" type="plane" size="10 10 0.1" material="matplane"/>

            <!-- World frame (origin) - larger than other frames -->
            <site name="world_origin" pos="0 0 0" size="0.02" rgba="0.5 0.5 0.5 0.3"/>

            <!-- ANNY Model (left side, at x=-0.5) with matched frame sites -->
            <body name="anny_body" pos="-0.5 0 1">
                <freejoint name="anny_joint"/>
                <geom name="anny_geom" type="mesh" mesh="anny_mesh" material="anny_material" rgba="0.6 0.8 0.7 0.15"/>
{anny_sites_xml}
            </body>

            <!-- SkeletonTorque Model (right side, at x=0.5) -->
            <body name="skeleton_offset" pos="0.5 0 0">
                <freejoint name="skel_root_joint"/>
{skeleton_body_str}
            </body>
        </worldbody>
    </mujoco>
    """

    with open(xml_path, 'w') as f:
        f.write(xml_content)

    print(f"Created combined XML with matched frames at {xml_path}")
    print(f"\nMatched anatomical landmarks:")
    for landmark_name in anatomical_matches.keys():
        print(f"  - {landmark_name}")

    return xml_path


def main():
    """
    Display both models with only matched anatomical frames.
    """
    print("=" * 80)
    print("ANNY + SkeletonTorque: Matched Anatomical Frames Display")
    print("=" * 80)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nTemporary directory: {temp_dir}")

    # Extract matched bone/body names from mapping
    anny_matched_bones = [anny_bone for anny_bone, _ in ANATOMICAL_MATCHES.values()]
    skeleton_matched_bodies = [skel_body for _, skel_body in ANATOMICAL_MATCHES.values()]

    # Create ANNY model
    print("\n" + "-" * 80)
    print("Loading ANNY Model...")
    print("-" * 80)
    anny_model = anny.create_fullbody_model(
        eyes=True,
        tongue=False,
        remove_unattached_vertices=True
    ).to(dtype=torch.float32, device='cpu')

    print(f"ANNY model created successfully!")

    # Generate ANNY mesh and matched bone poses
    anny_vertices, anny_faces, anny_matched_poses = create_anny_mesh_and_matched_bones(
        anny_model, anny_matched_bones
    )

    # Create SkeletonTorque environment
    print("\n" + "-" * 80)
    print("Loading SkeletonTorque Model...")
    print("-" * 80)
    skeleton_env = SkeletonTorque()
    print(f"SkeletonTorque model created successfully!")

    # Get skeleton matched body positions
    skeleton_matched_positions = get_skeleton_body_positions(
        skeleton_env, skeleton_matched_bodies
    )

    skeleton_env.stop()

    # Create combined XML
    print("\n" + "-" * 80)
    print("Creating Combined Scene with Matched Frames...")
    print("-" * 80)

    try:
        xml_path = create_combined_xml_with_matched_frames(
            anny_vertices, anny_faces, anny_matched_poses, anny_matched_bones,
            skeleton_matched_positions, skeleton_matched_bodies,
            ANATOMICAL_MATCHES, temp_dir
        )
    except Exception as e:
        print(f"Error creating combined XML: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return

    # Load combined model
    print("\nLoading combined MuJoCo model...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return

    print("\n" + "=" * 80)
    print("Viewer Controls:")
    print("  - Press E to toggle reference frames")
    print("  - Press T to make models transparent")
    print("  - Press H to hide/show the menu")
    print("  - Press TAB to switch cameras")
    print("  - Drag with LEFT mouse to rotate view")
    print("  - Drag with RIGHT mouse to move view")
    print("  - Scroll to zoom")
    print("  - Press ESC or close window to exit")
    print("=" * 80)
    print("\nDisplaying matched anatomical frames:")
    print("  - ANNY (left, transparent): Red frame sites at matched bones")
    print("  - SkeletonTorque (right): Body frames at matched locations")
    print("  - Models are 1 meter apart")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable frame visualization for sites (ANNY) and bodies (Skeleton)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE | mujoco.mjtFrame.mjFRAME_BODY

        # Make the world origin frame larger
        world_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "world_origin")
        if world_site_id >= 0:
            # Increase size for larger frame visualization
            model.site_size[world_site_id][0] = 0.05

        # Set camera
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 5.0
        viewer.cam.lookat[:] = [0, 0, 1]

        print("\nMatched frames displayed! Compare anatomical landmarks between models.")
        print("World origin frame (larger, gray) visible at (0, 0, 0)")
        print("Note: Site spheres are small and semi-transparent")

        # Main loop
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()

    print("\nCleaning up...")
    try:
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up.")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
