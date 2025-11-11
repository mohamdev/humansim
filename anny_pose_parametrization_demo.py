import torch
import roma # A PyTorch library useful to deal with space transformations.
import anny # The main library for the Anny model.
import trimesh # For 3D mesh visualization.
import numpy as np

light_gray = np.array([200, 200, 200, 255], dtype=np.uint8)

# Instantiate the model.
# Remark: the first instantiation may take a while. Latter calls will be faster thanks to caching.
anny_model = anny.create_fullbody_model(eyes=True, tongue=False, remove_unattached_vertices=True).to(dtype=torch.float32, device='cpu')

# Some helper objects for visualization.
trimesh_scene_transform = roma.Rigid(linear=roma.euler_to_rotmat('x', [-90.], degrees=True), translation=None).to_homogeneous().cpu().numpy()

mesh_material = trimesh.visual.material.PBRMaterial(baseColorFactor=[0.6, 0.8, 0.7, 0.5],
                                                        metallicFactor=0.5,
                                                        doubleSided=False,
                                                        alphaMode='BLEND')

world_axis = trimesh.creation.axis(axis_length=1.)
axis = trimesh.creation.axis(axis_length=0.1)

batch_size = 1
bone_id = anny_model.bone_labels.index('shoulder01.L')
print(f"Bone 'shoulder01.L' has id {bone_id}.")


pose_parameters = {label: torch.eye(4)[None] for label in anny_model.bone_labels}
pose_parameters["shoulder01.L"] = roma.Rigid(roma.euler_to_rotmat("z", [30.], degrees=True), translation=None).to_homogeneous()[None]
output = anny_model(pose_parameters=pose_parameters)
mesh = trimesh.Trimesh(vertices=output['vertices'].squeeze(), faces=anny_model.faces, process=False)

# Set vertex colors with alpha for transparency
n_vertices = len(mesh.vertices)
vertex_colors = np.tile([153, 204, 178, 128], (n_vertices, 1))  # RGBA with alpha=128 (50% transparent)
mesh.visual.vertex_colors = vertex_colors

scene = trimesh.Scene()
scene.add_geometry(mesh)
scene.add_geometry(world_axis)
scene.add_geometry(axis, transform=output["bone_poses"].squeeze(0)[bone_id].cpu().numpy())
scene.apply_transform(trimesh_scene_transform)
scene.show()

