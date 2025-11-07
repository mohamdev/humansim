# HumanSim

HumanSim is a library for human motion reconstruction from sparse and noisy data based on policy learning using reinforcement and imitation learning built on top of [loco-mujoco2](https://github.com/robfiras/loco-mujoco).

## Features

Currently, HumanSim features:

- **Kinematic Skeleton Animation**: Visualize human skeletal models with full kinematic chains
- **LAFAN1 Dataset Support**: Work with motion capture data from the LAFAN1 dataset
- **MyoSkeleton Integration**: Leverage the biomechanically accurate MyoSkeleton model with 151 joints
- **Interactive Visualization**: Real-time 3D visualization with MuJoCo viewer
- **Kinematic Tree Analysis**: Tools for exploring and understanding model structure

HumanSim is under active development. Planned features include:

- Reinforcement learning training pipelines
- Imitation learning from motion capture data
- Custom reward functions for kinematic and dynamic plausibility
- Policy export and deployment tools

## Installation

### Prerequisites

- Conda with Python 3.10 or higher

### Step 1: Create a Conda Environment

```bash
conda create -n humansim python=3.10
conda activate humansim
```

### Step 2: Install loco-mujoco

HumanSim depends on a local version of loco-mujoco. Clone loco-mujoco at the root, then install it in editable mode:

```bash
pip install -e ./loco-mujoco
```

## Usage

HumanSim provides several demonstration scripts for visualizing and analyzing human motion.

### 1. Display Kinematic Tree

Display the hierarchical structure of a humanoid model, showing all bodies, joints, and their relationships.

```bash
python display_kinematic_tree.py
```

**Options:**

```bash
# Display kinematic tree structure
python display_kinematic_tree.py --env MyoSkeleton

### 2. Display MyoSkeleton in Neutral Pose

View the MyoSkeleton model in a static neutral pose with 3D coordinate frames.

```bash
python display_myoskeleton_physical.py
```

### 3. Display MyoSkeleton with Trajectory Animation

Animate the MyoSkeleton performing a squat motion from motion capture data. 

```bash
python display_myoskeleton_with_all_frames.py
```

## Dependencies

HumanSim relies on the following dependencies (installed via loco-mujoco):

- **mujoco** (3.2.7)
- **numpy** (<2.0)
- **jax, flax**: Neural network framework support

## Troubleshooting

### ModuleNotFoundError: No module named 'loco_mujoco'

If you encounter this error, reinstall loco-mujoco in editable mode:

```bash
pip uninstall loco-mujoco -y
pip install -e /path/to/humansim/loco-mujoco
```

### Model Not Found

If the MyoSkeleton model is missing, you may need to initialize it:

```bash
loco-mujoco-myomodel-init
```

This downloads the required biomechanical model files.


## License

HumanSim is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

HumanSim builds upon:
- [loco-mujoco2](https://github.com/robfiras/loco-mujoco) by Firas Al-Hafez
- [MuJoCo](https://mujoco.org/) physics engine by DeepMind
- MyoSkeleton biomechanical model
- LAFAN1 motion capture dataset

## Citation

If you use HumanSim in your research, please cite the underlying loco-mujoco library:

```bibtex
@software{loco_mujoco,
  author = {Al-Hafez, Firas},
  title = {loco-mujoco: Imitation Learning Benchmark for Complex Locomotion},
  url = {https://github.com/robfiras/loco-mujoco},
  year = {2024}
}
```
