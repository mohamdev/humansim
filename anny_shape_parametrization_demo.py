from IPython.display import Markdown, display
import torch
import roma # A PyTorch library useful to deal with space transformations.
import anny # The main library for the Anny model.
import trimesh # For 3D mesh visualization.
import numpy as np


light_gray = np.array([200, 200, 200, 255], dtype=np.uint8)
# Instantiate the model, with all shape parameters available.
# Remark: the first instantiation may take a while. Latter calls will be faster thanks to caching.
anny_model = anny.create_fullbody_model(eyes=True, tongue=False, all_phenotypes=True, local_changes=True, remove_unattached_vertices=True)
# Use 32bit floating point precision on the CPU for this demo.
dtype = torch.float32
device = torch.device('cpu')
anny_model = anny_model.to(device=device, dtype=dtype)

# A simple transform to get a better view angle in 3D mesh visualizations.
trimesh_scene_transform = roma.Rigid(linear=roma.euler_to_rotmat('x', [-90.], degrees=True), translation=None).to_homogeneous().cpu().numpy()

print(f"{anny_model.template_vertices.shape[0]} vertices -- "
      f"{anny_model.faces.shape[0]} faces composed of {anny_model.faces.shape[1]} vertices each.")

print("List of phenotype parameters: "
      + ", ".join([label for label in anny_model.phenotype_labels]))

trimesh.Trimesh(vertices=anny_model.template_vertices.cpu().numpy(), vertex_colors=np.tile(light_gray, (anny_model.template_vertices.shape[0], 1)), faces=anny_model.faces.cpu().numpy()).apply_transform(trimesh_scene_transform).show()

batch_size = 5  # We can process multiple bodies at once in a batch. 

phenotype_kwargs = {key : torch.full((batch_size,), fill_value=0.5, dtype=dtype, device=device) for key in anny_model.phenotype_labels}
phenotype_kwargs['age'] = torch.linspace(0., 1., batch_size, dtype=dtype, device=device) # Example: vary the age parameter across the batch.
output = anny_model(phenotype_kwargs=phenotype_kwargs)

scene = trimesh.Scene()
for i in range(batch_size):
    # Create a mesh for each body in the batch.
    mesh = trimesh.Trimesh(vertices=output['vertices'][i].squeeze().cpu().numpy(), vertex_colors=np.tile(light_gray, (anny_model.template_vertices.shape[0], 1)), 
                            faces=anny_model.faces.cpu().numpy())
    transform = roma.Rigid(linear=None, translation=torch.tensor([i * 1., 0., 0.], dtype=dtype, device=device)).to_homogeneous().cpu().numpy()
    scene.add_geometry(mesh, transform=transform)

scene.apply_transform(trimesh_scene_transform)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the bodies in it.



import anny.shape_distribution
import anny.anthropometry
import matplotlib.pyplot as plt

phenotype_distribution = anny.shape_distribution.SimpleShapeDistribution(anny_model,
            morphological_age_distribution=torch.distributions.Uniform(low=0.0, high=60.0))

real_age, phenotype_kwargs = phenotype_distribution.sample(batch_size=200)
output = anny_model(phenotype_kwargs=phenotype_kwargs)

scene = trimesh.Scene()
i = -1
for u in range(4):
    for v in range(5):
        i += 1
        assert i < output['vertices'].shape[0], "Batch size is too small for the grid."
        # Create a mesh for each face in the batch.
        mesh = trimesh.Trimesh(vertices=output['vertices'][i].squeeze().cpu().numpy(), vertex_colors=np.tile(light_gray, (anny_model.template_vertices.shape[0], 1)), faces=anny_model.faces.cpu().numpy())
        transform = roma.Rigid(linear=None, translation=torch.tensor([v * 1., u * 1., 0.], dtype=dtype, device=device)).to_homogeneous().cpu().numpy()
        scene.add_geometry(mesh, transform=transform)
scene.apply_transform(trimesh_scene_transform)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the faces in

