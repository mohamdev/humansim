#!/usr/bin/env python3
"""
MuJoCo scene builder for matched anatomical visualization.

This module provides a class-based interface for creating combined MuJoCo scenes
that merge ANNY (with deformable skin) and SkeletonTorque models with matched
anatomical landmarks.
"""

import os
import tempfile
import shutil
import re
from typing import Dict, Tuple, Optional, List
import xml.etree.ElementTree as ET

import numpy as np
import torch
import mujoco
import anny
from anny.models.phenotype import RiggedModelWithPhenotypeParameters
from loco_mujoco.environments.humanoids import SkeletonTorque


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].

    Args:
        R: 3x3 rotation matrix

    Returns:
        quaternion as [w, x, y, z]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def transform_to_local(child_pose: np.ndarray, parent_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local transform (pos, quat) of child relative to parent.

    Args:
        child_pose: 4x4 transformation matrix of child in world frame
        parent_pose: 4x4 transformation matrix of parent in world frame

    Returns:
        (pos, quat) where pos is 3D position and quat is [w, x, y, z]
    """
    # Compute local transform: T_local = inv(T_parent) @ T_child
    parent_inv = np.linalg.inv(parent_pose)
    local_transform = parent_inv @ child_pose

    # Extract position
    pos = local_transform[:3, 3]

    # Extract rotation and convert to quaternion
    rot = local_transform[:3, :3]
    quat = rotation_matrix_to_quaternion(rot)

    return pos, quat


# Default anatomical landmark mappings between ANNY and SkeletonTorque
DEFAULT_ANATOMICAL_MATCHES = {
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


class MatchedSceneBuilder:
    """
    Builder class for creating MuJoCo scenes with matched ANNY and SkeletonTorque models.

    This class handles:
    - ANNY full-body model as deformable skin bound to articulated skeleton
    - SkeletonTorque model integration with name collision resolution
    - Matched anatomical landmark sites
    - Auxiliary visuals (floor, lights, materials)

    Example:
        >>> builder = MatchedSceneBuilder()
        >>> model, data = builder.build()
        >>> # Use model and data with MuJoCo viewer
    """

    def __init__(
        self,
        anatomical_matches: Optional[Dict[str, Tuple[str, str]]] = None,
        anny_offset: Tuple[float, float, float] = (-0.5, 0, 1),
        skeleton_offset: Tuple[float, float, float] = (0.5, 0, 0),
        skin_opacity: float = 0.15,
        frame_size: float = 0.3,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize the scene builder.

        Args:
            anatomical_matches: Dictionary mapping landmark names to (anny_bone, skel_body) tuples.
                               If None, uses DEFAULT_ANATOMICAL_MATCHES.
            anny_offset: (x, y, z) offset for ANNY model placement
            skeleton_offset: (x, y, z) offset for SkeletonTorque model placement
            skin_opacity: Alpha value for ANNY skin material (0.0-1.0)
            frame_size: Length of coordinate frame visualization
            temp_dir: Directory for temporary files. If None, creates a temp directory.
        """
        self.anatomical_matches = anatomical_matches or DEFAULT_ANATOMICAL_MATCHES
        self.anny_offset = anny_offset
        self.skeleton_offset = skeleton_offset
        self.skin_opacity = skin_opacity
        self.frame_size = frame_size

        # Create temp directory if needed
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
            self._owns_temp_dir = True
        else:
            self.temp_dir = temp_dir
            self._owns_temp_dir = False
            os.makedirs(temp_dir, exist_ok=True)

        # Will be populated during build
        self.anny_model = None
        self.model = None
        self.data = None
        self.xml_path = None

    def __del__(self):
        """Cleanup temporary directory if we created it."""
        try:
            if self._owns_temp_dir and self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

    def build(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """
        Build the complete MuJoCo scene with both models.

        Returns:
            (model, data): MuJoCo model and data instances
        """
        print("=" * 80)
        print("Building Matched Scene (ANNY + SkeletonTorque)")
        print("=" * 80)

        # Load ANNY model
        print("\nLoading ANNY model...")
        self.anny_model = self._load_anny_model()

        # Generate ANNY geometry and bone poses
        print("Generating ANNY geometry in neutral pose...")
        anny_data = self._generate_anny_neutral_pose()

        # Get SkeletonTorque data
        print("\nLoading SkeletonTorque model...")
        skeleton_data = self._get_skeleton_data()

        # Create combined XML
        print("\nCreating combined MuJoCo XML...")
        self.xml_path = self._create_combined_xml(anny_data, skeleton_data)

        # Load MuJoCo model
        print("Loading MuJoCo model...")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        print("\n" + "=" * 80)
        print("Scene built successfully!")
        print(f"Temporary files in: {self.temp_dir}")
        print("=" * 80)

        return self.model, self.data

    def cleanup(self):
        """Manually cleanup temporary files."""
        if self._owns_temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # ========== Private Methods ==========

    def _load_anny_model(self) -> RiggedModelWithPhenotypeParameters:
        """Load and initialize ANNY model."""
        model = anny.create_fullbody_model(
            eyes=True,
            tongue=False,
            remove_unattached_vertices=True
        ).to(dtype=torch.float32, device='cpu')

        print(f"  Loaded ANNY with {len(model.bone_labels)} bones, "
              f"{model.faces.shape[0]} faces")
        return model

    def _generate_anny_neutral_pose(self) -> dict:
        """Generate ANNY mesh and bone poses in neutral configuration."""
        # Set all bones to identity (neutral pose)
        pose_parameters = {
            label: torch.eye(4, dtype=torch.float32)[None]
            for label in self.anny_model.bone_labels
        }

        # Generate mesh
        output = self.anny_model(pose_parameters=pose_parameters)

        vertices = output['vertices'].squeeze(0).cpu().numpy()
        bone_poses = output['bone_poses'].squeeze(0).cpu().numpy()
        faces = self.anny_model.faces.cpu().numpy()

        # Get skinning data
        bone_weights = self.anny_model.vertex_bone_weights.cpu().numpy()
        bone_indices = self.anny_model.vertex_bone_indices.cpu().numpy()

        # Extract matched bone poses
        matched_bone_poses = {}
        for anny_bone, _ in self.anatomical_matches.values():
            if anny_bone in self.anny_model.bone_labels:
                bone_idx = self.anny_model.bone_labels.index(anny_bone)
                matched_bone_poses[anny_bone] = bone_poses[bone_idx]

        print(f"  Generated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        print(f"  Extracted {len(matched_bone_poses)} matched bone poses")

        return {
            'vertices': vertices,
            'faces': faces,
            'bone_poses': bone_poses,
            'matched_bone_poses': matched_bone_poses,
            'bone_weights': bone_weights,
            'bone_indices': bone_indices,
            'bone_labels': self.anny_model.bone_labels,
            'bone_parents': self.anny_model.bone_parents,
        }

    def _get_skeleton_data(self) -> dict:
        """Load SkeletonTorque and extract required data."""
        skeleton_env = SkeletonTorque()
        skeleton_env.reset()
        mujoco.mj_forward(skeleton_env._model, skeleton_env._data)

        # Get matched body positions
        matched_body_positions = {}
        for _, skeleton_body in self.anatomical_matches.values():
            body_id = mujoco.mj_name2id(
                skeleton_env._model,
                mujoco.mjtObj.mjOBJ_BODY,
                skeleton_body
            )
            if body_id >= 0:
                matched_body_positions[skeleton_body] = skeleton_env._data.xpos[body_id].copy()

        # Get XML path
        xml_path = self._get_skeleton_xml_path()

        skeleton_env.stop()

        print(f"  Loaded SkeletonTorque from {xml_path}")
        print(f"  Extracted {len(matched_body_positions)} matched body positions")

        return {
            'xml_path': xml_path,
            'matched_positions': matched_body_positions,
        }

    def _get_skeleton_xml_path(self) -> str:
        """Get path to SkeletonTorque XML file."""
        local_path = "./loco-mujoco/loco_mujoco/models/skeleton/skeleton_torque.xml"
        if os.path.exists(local_path):
            return os.path.abspath(local_path)

        env = SkeletonTorque()
        xml_path = env._model_path
        env.stop()
        return xml_path

    def _create_combined_xml(self, anny_data: dict, skeleton_data: dict) -> str:
        """Create the combined MuJoCo XML file."""
        xml_path = os.path.join(self.temp_dir, "matched_scene.xml")

        # Generate XML sections
        anny_skeleton_xml = self._generate_anny_skeleton_xml(anny_data)
        anny_skin_xml = self._generate_anny_skin_xml(anny_data)

        skeleton_xml_parts = self._process_skeleton_xml(skeleton_data['xml_path'])

        # Build complete XML
        xml_content = self._assemble_xml(
            anny_skeleton_xml,
            anny_skin_xml,
            skeleton_xml_parts
        )

        with open(xml_path, 'w') as f:
            f.write(xml_content)

        print(f"  Created combined XML: {xml_path}")
        return xml_path

    def _generate_anny_skeleton_xml(self, anny_data: dict) -> str:
        """Generate MuJoCo body hierarchy for ANNY skeleton with proper local transforms and joints."""
        bone_labels = anny_data['bone_labels']
        bone_parents = anny_data['bone_parents']
        bone_poses = anny_data['bone_poses']
        matched_bone_poses = anny_data['matched_bone_poses']

        # Build site attachments for matched landmarks
        bone_to_sites = {}
        for landmark_name, (anny_bone, _) in self.anatomical_matches.items():
            if anny_bone in bone_labels:
                safe_name = landmark_name.replace(' ', '_')
                site_xml = f'<site name="anny_{safe_name}" pos="0 0 0" size="0.005" rgba="1 0 0 0.2"/>'
                bone_to_sites[anny_bone] = bone_to_sites.get(anny_bone, []) + [site_xml]

        # Build hierarchy recursively
        def build_body_xml(bone_idx: int, indent: int = 4) -> str:
            bone_name = bone_labels[bone_idx]
            safe_name = bone_name.replace('.', '_')
            bone_pose = bone_poses[bone_idx]

            # Get parent transform
            parent_idx = bone_parents[bone_idx]
            if parent_idx >= 0:
                parent_pose = bone_poses[parent_idx]
                pos, quat = transform_to_local(bone_pose, parent_pose)
            else:
                # Root bone - use world transform
                pos = bone_pose[:3, 3]
                quat = rotation_matrix_to_quaternion(bone_pose[:3, :3])

            ind = ' ' * indent

            # Start body with local transform
            xml = f'{ind}<body name="anny_{safe_name}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '\
                  f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}">\n'

            # Add joint for non-root bones (ball joint for full 3DOF rotation)
            if parent_idx >= 0:
                xml += f'{ind}    <joint name="anny_{safe_name}_joint" type="ball" damping="0.5" '\
                       f'stiffness="0" limited="false"/>\n'

            # Add a small geom for visualization
            xml += f'{ind}    <geom name="anny_{safe_name}_marker" type="sphere" size="0.005" '\
                   f'rgba="0.8 0.2 0.2 0.3" contype="0" conaffinity="0"/>\n'

            # Add landmark sites attached to this bone
            if bone_name in bone_to_sites:
                for site_xml in bone_to_sites[bone_name]:
                    xml += f'{ind}    {site_xml}\n'

            # Find and add children
            for child_idx, child_parent_idx in enumerate(bone_parents):
                if child_parent_idx == bone_idx:
                    xml += build_body_xml(child_idx, indent + 4)

            xml += f'{ind}</body>\n'
            return xml

        # Find root bone(s) - those with parent_idx == -1
        root_bones = [i for i, p in enumerate(bone_parents) if p == -1]

        skeleton_xml = ""
        for root_idx in root_bones:
            skeleton_xml += build_body_xml(root_idx, indent=16)

        return skeleton_xml

    def _generate_anny_skin_xml(self, anny_data: dict) -> str:
        """Generate MuJoCo skin element for ANNY mesh."""
        vertices = anny_data['vertices']
        faces = anny_data['faces']
        bone_weights = anny_data['bone_weights']
        bone_indices = anny_data['bone_indices']
        bone_labels = anny_data['bone_labels']
        bone_poses = anny_data['bone_poses']

        # Convert quad faces to triangles if needed
        if faces.shape[1] == 4:
            triangles = []
            for face in faces:
                triangles.append([face[0], face[1], face[2]])
                triangles.append([face[0], face[2], face[3]])
            faces = np.array(triangles)

        # Build vertex attribute string (space-separated x y z coordinates)
        vertex_str = '  '.join([f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in vertices])

        # Build texcoord attribute (dummy UVs for now - MuJoCo requires them)
        texcoord_str = '  '.join([f'{i/(len(vertices)-1):.4f} 0' for i in range(len(vertices))])

        # Build face attribute string (space-separated vertex indices)
        face_str = '  '.join([f'{f[0]} {f[1]} {f[2]}' for f in faces])

        # Start skin XML (under asset section)
        skin_xml = f'        <skin name="anny_skin" '\
                   f'material="anny_skin_material" '\
                   f'rgba="0.6 0.8 0.7 {self.skin_opacity}" '\
                   f'vertex="{vertex_str}" '\
                   f'texcoord="{texcoord_str}" '\
                   f'face="{face_str}">\n'

        # Add bone elements with binding data
        for bone_idx, label in enumerate(bone_labels):
            safe_name = label.replace('.', '_')
            pose = bone_poses[bone_idx]

            # Extract position and rotation from bone rest pose matrix
            bindpos = pose[:3, 3]
            bindrot = pose[:3, :3]
            bindquat = rotation_matrix_to_quaternion(bindrot)

            # Find vertices influenced by this bone
            vertex_mask = bone_indices == bone_idx
            influenced_verts = np.where(vertex_mask.any(axis=1))[0]

            if len(influenced_verts) == 0:
                continue

            # Get vertex IDs and weights for this bone
            vertid_list = []
            vertweight_list = []

            for vert_idx in influenced_verts:
                # Find which slot this bone occupies for this vertex
                bone_slots = np.where(bone_indices[vert_idx] == bone_idx)[0]
                if len(bone_slots) > 0:
                    weight = bone_weights[vert_idx, bone_slots[0]]
                    if weight > 1e-6:
                        vertid_list.append(str(vert_idx))
                        vertweight_list.append(f'{weight:.6f}')

            if len(vertid_list) == 0:
                continue

            vertid_str = ' '.join(vertid_list)
            vertweight_str = ' '.join(vertweight_list)

            skin_xml += f'            <bone body="anny_{safe_name}" '\
                       f'bindpos="{bindpos[0]:.6f} {bindpos[1]:.6f} {bindpos[2]:.6f}" '\
                       f'bindquat="{bindquat[0]:.6f} {bindquat[1]:.6f} {bindquat[2]:.6f} {bindquat[3]:.6f}" '\
                       f'vertid="{vertid_str}" '\
                       f'vertweight="{vertweight_str}"/>\n'

        skin_xml += '        </skin>\n'

        return skin_xml

    def _process_skeleton_xml(self, skeleton_xml_path: str) -> dict:
        """Process SkeletonTorque XML and add name prefixes."""
        tree = ET.parse(skeleton_xml_path)
        root = tree.getroot()

        skeleton_meshdir = os.path.dirname(skeleton_xml_path)

        # Extract defaults
        skeleton_defaults = root.find('default')
        defaults_str = ""
        if skeleton_defaults is not None:
            for child in skeleton_defaults:
                default_str = ET.tostring(child, encoding='unicode')
                defaults_str += "            " + default_str + "\n"

        # Extract and process assets
        skeleton_assets = root.find('asset')
        assets_str = ""
        if skeleton_assets is not None:
            for child in skeleton_assets:
                asset_str = ET.tostring(child, encoding='unicode')
                asset_str = asset_str.replace('name="', 'name="skel_')
                asset_str = asset_str.replace('mesh="', 'mesh="skel_')
                if 'file="' in asset_str and not asset_str.split('file="')[1].startswith('/'):
                    asset_str = asset_str.replace('file="', f'file="{skeleton_meshdir}/')
                assets_str += "            " + asset_str + "\n"

        # Extract and process worldbody
        skeleton_worldbody = root.find('worldbody')
        skeleton_body = skeleton_worldbody.find('body')

        body_str = ET.tostring(skeleton_body, encoding='unicode')
        body_str = body_str.replace('name="', 'name="skel_')
        body_str = body_str.replace('joint="', 'joint="skel_')
        body_str = body_str.replace('mesh="', 'mesh="skel_')
        body_str = body_str.replace(' class="mimic"', '')

        # Remove original freejoint
        body_str = re.sub(r'<freejoint[^/]*/>', '', body_str)
        body_str = re.sub(r'<freejoint[^>]*>[^<]*</freejoint>', '', body_str)

        return {
            'defaults': defaults_str,
            'assets': assets_str,
            'body': body_str,
        }

    def _assemble_xml(
        self,
        anny_skeleton_xml: str,
        anny_skin_xml: str,
        skeleton_parts: dict
    ) -> str:
        """Assemble all parts into complete XML."""

        xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="matched_scene">
    <compiler angle="radian" meshdir="{self.temp_dir}" balanceinertia="true" inertiafromgeom="auto"/>

    <option timestep="0.01" gravity="0 0 -9.81">
        <flag contact="disable"/>
    </option>

    <default>
{skeleton_parts['defaults']}
    </default>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <map force="0.1" fogstart="5" fogend="10"/>
        <scale framewidth="0.015" framelength="{self.frame_size}"/>
    </visual>

    <asset>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".3 .4 .5" width="512" height="512"/>
        <texture name="texgeom" type="2d" builtin="flat" mark="cross" width="128" height="128" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        <material name="anny_skin_material" rgba="0.6 0.8 0.7 {self.skin_opacity}"/>

{skeleton_parts['assets']}

{anny_skin_xml}
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5" dir="0 0 -1"/>
        <light directional="true" diffuse=".4 .4 .4" pos="0 0 3" dir="0 1 -1"/>

        <geom name="floor" type="plane" size="10 10 0.1" material="matplane"/>

        <!-- World frame (origin) -->
        <site name="world_origin" pos="0 0 0" size="0.02" rgba="0.5 0.5 0.5 0.3"/>

        <!-- ANNY Model with deformable skin -->
        <body name="anny_offset_body" pos="{self.anny_offset[0]} {self.anny_offset[1]} {self.anny_offset[2]}">
            <freejoint name="anny_freejoint"/>
{anny_skeleton_xml}
        </body>

        <!-- SkeletonTorque Model -->
        <body name="skeleton_offset" pos="{self.skeleton_offset[0]} {self.skeleton_offset[1]} {self.skeleton_offset[2]}">
            <freejoint name="skel_root_joint"/>
{skeleton_parts['body']}
        </body>
    </worldbody>

</mujoco>
"""
        return xml_content


def main():
    """Example usage of MatchedSceneBuilder."""
    builder = MatchedSceneBuilder()
    model, data = builder.build()

    print("\nScene components:")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  Geoms: {model.ngeom}")
    if model.nskin > 0:
        print(f"  Skins: {model.nskin}")
        print(f"  Skin vertices: {model.skin_vertnum[0] if model.nskin > 0 else 0}")

    return builder, model, data


if __name__ == "__main__":
    main()
