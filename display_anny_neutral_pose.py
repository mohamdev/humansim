#!/usr/bin/env python3
"""
Script to display the ANNY parametric human model in neutral pose using MuJoCo viewer.

ANNY is a free and interpretable human body model for all ages, written in PyTorch.
This script displays the model in its neutral/rest pose with coordinate frames.
"""

import torch
import numpy as np
import mujoco
import mujoco.viewer
import anny
import tempfile
import os


def create_anny_mesh_in_neutral_pose(anny_model):
    """
    Generate ANNY mesh in neutral pose (all bones at identity).

    Args:
        anny_model: The ANNY model instance

    Returns:
        vertices: numpy array of shape (n_vertices, 3)
        faces: numpy array of shape (n_faces, 3 or 4)
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

    print(f"Generated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"Vertex range: x=[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}], "
          f"y=[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}], "
          f"z=[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

    return vertices, faces


def create_mujoco_xml_with_mesh(vertices, faces):
    """
    Create a MuJoCo XML model with the ANNY mesh.

    Args:
        vertices: numpy array of shape (n_vertices, 3)
        faces: numpy array of shape (n_faces, 3 or 4)

    Returns:
        xml_path: Path to the created XML file
        mesh_path: Path to the mesh STL file
    """
    # Create temporary directory for the mesh and XML
    temp_dir = tempfile.mkdtemp()
    mesh_path = os.path.join(temp_dir, "anny_mesh.stl")
    xml_path = os.path.join(temp_dir, "anny_model.xml")

    # Save mesh as STL (MuJoCo can load STL files)
    # We need to convert faces to triangles if they are quads
    if faces.shape[1] == 4:
        # Convert quads to triangles
        print("Converting quad faces to triangles...")
        triangles = []
        for face in faces:
            # Split quad into two triangles
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        faces_tri = np.array(triangles)
    else:
        faces_tri = faces

    # Create STL mesh using trimesh
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces_tri, process=False)
        mesh.export(mesh_path)
        print(f"Saved mesh to {mesh_path}")
    except ImportError:
        print("Warning: trimesh not available, using basic STL export")
        # Fallback: write STL manually
        write_stl(mesh_path, vertices, faces_tri)

    # Create MuJoCo XML
    xml_content = f"""
    <mujoco model="anny_neutral">
        <compiler angle="radian" meshdir="{temp_dir}"/>

        <option timestep="0.01" gravity="0 0 -9.81">
            <flag contact="disable"/>
        </option>

        <visual>
            <global offwidth="1920" offheight="1080"/>
            <quality shadowsize="4096"/>
            <map force="0.1" fogstart="3" fogend="5"/>
            <scale framewidth="0.005" framelength="0.5"/>
        </visual>

        <asset>
            <mesh name="anny_mesh" file="anny_mesh.stl" scale="1 1 1"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".3 .4 .5" width="512" height="512"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            <material name="anny_material" rgba="0.6 0.8 0.7 1.0"/>
        </asset>

        <worldbody>
            <light directional="true" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5" dir="0 0 -1"/>
            <light directional="true" diffuse=".4 .4 .4" pos="0 0 3" dir="0 1 -1"/>

            <geom name="floor" type="plane" size="5 5 0.1" material="matplane"/>

            <body name="anny_body" pos="0 0 1">
                <freejoint/>
                <geom name="anny_geom" type="mesh" mesh="anny_mesh" material="anny_material"/>
                <site name="root" pos="0 0 0" size="0.01"/>
            </body>
        </worldbody>
    </mujoco>
    """

    with open(xml_path, 'w') as f:
        f.write(xml_content)

    print(f"Created MuJoCo XML at {xml_path}")

    return xml_path, temp_dir


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


def main():
    """
    Display ANNY model in neutral pose using MuJoCo viewer.
    """
    print("=" * 60)
    print("ANNY Parametric Human Model - Neutral Pose Display")
    print("=" * 60)

    # Create ANNY model
    print("\nInstantiating ANNY model...")
    print("(First instantiation may take a while due to caching)")

    anny_model = anny.create_fullbody_model(
        eyes=True,
        tongue=False,
        remove_unattached_vertices=True
    ).to(dtype=torch.float32, device='cpu')

    print(f"ANNY model created successfully!")
    print(f"Number of bones: {anny_model.bone_count}")
    print(f"Bone labels: {len(anny_model.bone_labels)}")

    # Generate mesh in neutral pose
    vertices, faces = create_anny_mesh_in_neutral_pose(anny_model)

    # Create MuJoCo model with the mesh
    xml_path, temp_dir = create_mujoco_xml_with_mesh(vertices, faces)

    # Load the MuJoCo model
    print("\nLoading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Reset to initial state
    mujoco.mj_resetData(model, data)

    print("\n" + "=" * 60)
    print("Viewer Controls:")
    print("  - Press E to toggle reference frames (coordinate axes)")
    print("  - Press T to make the model transparent")
    print("  - Press H to hide/show the menu")
    print("  - Press TAB to switch cameras")
    print("  - Drag with LEFT mouse button to rotate view")
    print("  - Drag with RIGHT mouse button to move view")
    print("  - Scroll to zoom")
    print("  - Press ESC or close window to exit")
    print("=" * 60)

    # Launch interactive viewer
    print("\nLaunching MuJoCo viewer...")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable frame visualization
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        # Set camera for better view
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 1]

        # Main loop - just keep the viewer open
        while viewer.is_running():
            # Compute forward kinematics
            mujoco.mj_forward(model, data)

            # Sync viewer
            viewer.sync()

    print("\nCleaning up...")
    # Clean up temporary files
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up.")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
