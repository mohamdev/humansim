#!/usr/bin/env python3
"""
Script to display both ANNY and SkeletonTorque models side by side in neutral pose.

This script displays:
- ANNY parametric human model (left side) with transparent mesh and bone frames
- SkeletonTorque model (right side) with body frames
- Both models positioned 1 meter apart on the same plane
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


def create_anny_mesh_in_neutral_pose(anny_model):
    """
    Generate ANNY mesh in neutral pose (all bones at identity).

    Args:
        anny_model: The ANNY model instance

    Returns:
        vertices: numpy array of shape (n_vertices, 3)
        faces: numpy array of shape (n_faces, 3 or 4)
        bone_poses: numpy array of shape (n_bones, 4, 4)
        bone_labels: list of bone names
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

    # Extract bone poses (4x4 transformation matrices)
    bone_poses = output['bone_poses'].squeeze(0).cpu().numpy()
    bone_labels = anny_model.bone_labels

    print(f"Generated ANNY mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"Number of ANNY bones: {len(bone_labels)}")

    return vertices, faces, bone_poses, bone_labels


def write_stl(filename, vertices, faces):
    """
    Write a simple binary STL file.
    """
    with open(filename, 'wb') as f:
        # Header
        f.write(b'\0' * 80)
        # Number of triangles
        f.write(np.uint32(len(faces)).tobytes())

        for face in faces:
            # Get triangle vertices
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Calculate normal
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-8)

            # Write normal
            f.write(normal.astype(np.float32).tobytes())
            # Write vertices
            f.write(v0.astype(np.float32).tobytes())
            f.write(v1.astype(np.float32).tobytes())
            f.write(v2.astype(np.float32).tobytes())
            # Attribute byte count
            f.write(np.uint16(0).tobytes())


def get_skeleton_xml_path():
    """
    Get the path to the SkeletonTorque XML file.

    Returns:
        Path to skeleton_torque.xml
    """
    # First try the local path
    local_path = "./loco-mujoco/loco_mujoco/models/skeleton/skeleton_torque.xml"
    if os.path.exists(local_path):
        return os.path.abspath(local_path)

    # Otherwise, try to get it from the environment
    env = SkeletonTorque()
    xml_path = env._model_path
    env.stop()
    return xml_path


def create_combined_xml(anny_vertices, anny_faces, anny_bone_poses, anny_bone_labels, temp_dir):
    """
    Create a combined MuJoCo XML with both ANNY and SkeletonTorque models.

    Args:
        anny_vertices: ANNY mesh vertices
        anny_faces: ANNY mesh faces
        anny_bone_poses: ANNY bone transformation matrices
        anny_bone_labels: ANNY bone names
        temp_dir: Temporary directory for files

    Returns:
        xml_path: Path to the combined XML file
    """
    mesh_path = os.path.join(temp_dir, "anny_mesh.stl")
    xml_path = os.path.join(temp_dir, "combined_model.xml")
    skeleton_xml_path = os.path.join(temp_dir, "skeleton_only.xml")

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

    # Save ANNY mesh as STL
    import trimesh
    mesh = trimesh.Trimesh(vertices=anny_vertices, faces=faces_tri, process=False)
    mesh.export(mesh_path)
    print(f"Saved ANNY mesh to {mesh_path}")

    # Generate ANNY bone sites XML
    anny_bone_sites_xml = ""
    for i, (label, pose) in enumerate(zip(anny_bone_labels, anny_bone_poses)):
        pos = pose[:3, 3]
        safe_label = label.replace('.', '_').replace(' ', '_')
        anny_bone_sites_xml += f'                <site name="anny_bone_{safe_label}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" size="0.01"/>\n'

    print(f"Generated {len(anny_bone_labels)} ANNY bone sites")

    # Load SkeletonTorque model to verify it exists
    print("Loading SkeletonTorque model...")
    skeleton_env = SkeletonTorque()
    skeleton_env.stop()

    # Get the original skeleton XML path
    original_skeleton_xml_path = get_skeleton_xml_path()
    print(f"Using SkeletonTorque XML from: {original_skeleton_xml_path}")

    # Create combined XML with manual placement
    # Read skeleton XML to get mesh paths
    import xml.etree.ElementTree as ET
    skeleton_tree = ET.parse(original_skeleton_xml_path)
    skeleton_root = skeleton_tree.getroot()

    # Extract worldbody content from skeleton
    skeleton_worldbody = skeleton_root.find('worldbody')
    if skeleton_worldbody is None:
        print("Error: Could not find worldbody in skeleton XML")
        return None

    # Get the first body (pelvis) from skeleton worldbody
    skeleton_body = skeleton_worldbody.find('body')

    # Convert skeleton body to string and adjust names to avoid conflicts
    skeleton_body_str = ET.tostring(skeleton_body, encoding='unicode')
    # Add prefix to avoid name conflicts
    skeleton_body_str = skeleton_body_str.replace('name="', 'name="skel_')
    skeleton_body_str = skeleton_body_str.replace('joint="', 'joint="skel_')
    skeleton_body_str = skeleton_body_str.replace('mesh="', 'mesh="skel_')
    # Remove mimic class references (they come from included files we don't have)
    skeleton_body_str = skeleton_body_str.replace(' class="mimic"', '')
    skeleton_body_str = skeleton_body_str.replace(' class="skel_mimic"', '')
    # Remove the freejoint from the pelvis body (it can't be nested)
    import re
    skeleton_body_str = re.sub(r'<freejoint[^/]*/>',  '', skeleton_body_str)
    skeleton_body_str = re.sub(r'<freejoint[^>]*>[^<]*</freejoint>', '', skeleton_body_str)

    # Get skeleton model directory for mesh paths
    skeleton_meshdir = os.path.dirname(get_skeleton_xml_path())

    # Extract default class definitions from skeleton
    skeleton_defaults = skeleton_root.find('default')
    skeleton_defaults_str = ""
    if skeleton_defaults is not None:
        for child in skeleton_defaults:
            default_str = ET.tostring(child, encoding='unicode')
            skeleton_defaults_str += "            " + default_str + "\n"

    # Extract asset definitions from skeleton
    skeleton_assets = skeleton_root.find('asset')
    skeleton_assets_str = ""
    if skeleton_assets is not None:
        for child in skeleton_assets:
            asset_str = ET.tostring(child, encoding='unicode')
            # Prefix asset names
            asset_str = asset_str.replace('name="', 'name="skel_')
            asset_str = asset_str.replace('mesh="', 'mesh="skel_')
            # Fix file paths to be absolute
            if 'file="' in asset_str and not asset_str.split('file="')[1].startswith('/'):
                # Make relative paths absolute
                asset_str = asset_str.replace('file="', f'file="{skeleton_meshdir}/')
            skeleton_assets_str += "            " + asset_str + "\n"

    # Create combined XML
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="combined_models">
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
            <scale framewidth="0.0025" framelength="0.083"/>
        </visual>

        <asset>
            <mesh name="anny_mesh" file="anny_mesh.stl" scale="1 1 1"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".3 .4 .5" width="512" height="512"/>
            <texture name="texgeom" type="2d" builtin="flat" mark="cross" width="128" height="128" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            <material name="anny_material" rgba="0.6 0.8 0.7 0.3"/>

{skeleton_assets_str}
        </asset>

        <worldbody>
            <light directional="true" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5" dir="0 0 -1"/>
            <light directional="true" diffuse=".4 .4 .4" pos="0 0 3" dir="0 1 -1"/>

            <geom name="floor" type="plane" size="10 10 0.1" material="matplane"/>

            <!-- ANNY Model (left side, at x=-0.5) -->
            <body name="anny_body" pos="-0.5 0 1">
                <freejoint name="anny_joint"/>
                <geom name="anny_geom" type="mesh" mesh="anny_mesh" material="anny_material"/>
{anny_bone_sites_xml}
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

    print(f"Created combined XML at {xml_path}")

    return xml_path


def main():
    """
    Display both ANNY and SkeletonTorque models side by side in neutral pose.
    """
    print("=" * 80)
    print("Combined Display: ANNY + SkeletonTorque in Neutral Pose")
    print("=" * 80)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nTemporary directory: {temp_dir}")

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
    print(f"Number of bones: {anny_model.bone_count}")

    # Generate ANNY mesh
    anny_vertices, anny_faces, anny_bone_poses, anny_bone_labels = create_anny_mesh_in_neutral_pose(anny_model)

    # Create SkeletonTorque to verify it's available
    print("\n" + "-" * 80)
    print("Loading SkeletonTorque Model...")
    print("-" * 80)
    try:
        skeleton_env = SkeletonTorque()
        print(f"SkeletonTorque model created successfully!")
        print(f"Number of bodies: {skeleton_env._model.nbody}")
        skeleton_env.stop()
    except Exception as e:
        print(f"Error creating SkeletonTorque: {e}")
        shutil.rmtree(temp_dir)
        return

    # Create combined XML
    print("\n" + "-" * 80)
    print("Creating Combined MuJoCo Scene...")
    print("-" * 80)

    try:
        xml_path = create_combined_xml(
            anny_vertices, anny_faces, anny_bone_poses, anny_bone_labels, temp_dir
        )
    except Exception as e:
        print(f"Error creating combined XML: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return

    # Load the combined MuJoCo model
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
    print("  - Press E to toggle reference frames (coordinate axes)")
    print("  - Press T to make models transparent")
    print("  - Press H to hide/show the menu")
    print("  - Press TAB to switch cameras")
    print("  - Drag with LEFT mouse button to rotate view")
    print("  - Drag with RIGHT mouse button to move view")
    print("  - Scroll to zoom")
    print("  - Press ESC or close window to exit")
    print("=" * 80)
    print("\nLaunching MuJoCo viewer...")
    print("Models are positioned 1 meter apart:")
    print("  - ANNY (transparent): left side")
    print("  - SkeletonTorque: right side")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable frame visualization for sites (ANNY bones) and bodies (SkeletonTorque)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE | mujoco.mjtFrame.mjFRAME_BODY

        # Set camera for better view (centered between both models)
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 5.0
        viewer.cam.lookat[:] = [0, 0, 1]

        print("\nBoth models displayed with coordinate frames!")

        # Main loop
        while viewer.is_running():
            # Compute forward kinematics
            mujoco.mj_forward(model, data)

            # Sync viewer
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
