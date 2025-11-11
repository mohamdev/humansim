# Skeleton-to-ANNY Calibration

This calibration script computes SE(3) alignments between the SkeletonTorque biomechanical model (ISB convention) and the ANNY parametric mesh model.

## Quick Start

```bash
# Run calibration with visualization
python calibrate_skeleton_to_anny.py

# Run without visualization (faster)
python calibrate_skeleton_to_anny.py --no-viz

# List all body pair mappings
python calibrate_skeleton_to_anny.py --list-bodies

# Custom output paths
python calibrate_skeleton_to_anny.py --save my_align.json my_joints.json
```

## Output Files

### `align_se3.json` (6.8 KB)
Contains SE(3) transformation matrices for each body pair:
- **rotation**: 3×3 rotation matrix from skeleton frame to ANNY frame
- **translation**: 3D translation vector
- **anny_bone**: corresponding ANNY bone label

Example:
```json
{
  "pelvis": {
    "rotation": [[1.0, 0.0, 0.0], ...],
    "translation": [0.0, 0.945, -0.238],
    "anny_bone": "root"
  }
}
```

### `joint_meta.json` (3.9 KB)
Contains metadata for all skeleton joints:
- **type**: `hinge`, `ball`, or `free`
- **axis**: rotation axis (for hinge joints)
- **sign**: coordinate mapping sign (±1)
- **offset**: joint angle offset

## Body Mappings (14 pairs)

| Skeleton Body | ANNY Bone       | Description        |
|---------------|-----------------|-------------------|
| pelvis        | root            | Root/pelvis       |
| torso         | spine03         | Upper spine       |
| femur_r       | upperleg01.R    | Right thigh       |
| femur_l       | upperleg01.L    | Left thigh        |
| tibia_r       | lowerleg01.R    | Right shin        |
| tibia_l       | lowerleg01.L    | Left shin         |
| talus_r       | foot.R          | Right foot        |
| talus_l       | foot.L          | Left foot         |
| humerus_r     | shoulder01.R    | Right upper arm   |
| humerus_l     | shoulder01.L    | Left upper arm    |
| ulna_r        | lowerarm01.R    | Right forearm     |
| ulna_l        | lowerarm01.L    | Left forearm      |
| hand_r        | wrist.R         | Right hand        |
| hand_l        | wrist.L         | Left hand         |

## Customization

Edit the `MANUAL_NAME_MAP` dictionary at the top of `calibrate_skeleton_to_anny.py`:

```python
MANUAL_NAME_MAP = {
    'pelvis': 'root',
    'torso': 'spine03',
    # Add or modify mappings here
}
```

## Viewer Controls (when visualization enabled)

- **E**: Toggle reference frames
- **T**: Toggle transparency
- **H**: Hide/show menu
- **Left mouse drag**: Rotate view
- **Right mouse drag**: Pan view
- **Scroll wheel**: Zoom
- **ESC**: Exit viewer

## Accuracy

The script computes and displays RMS errors:
- **RMS rotation error**: Should be near 0.00 degrees
- **RMS translation error**: Should be near 0.00 cm

These errors measure how well the alignment transforms fit the neutral poses.

## Technical Details

### Coordinate Systems

**ISB (Skeleton):**
- X: Antero-posterior (back → front)
- Y: Longitudinal (downward → upward)
- Z: Medio-lateral (middle → right)

**ANNY:**
- X: Medio-lateral
- Y: Postero-anterior
- Z: Upward (vertical)

### SE(3) Alignment Formula

For each body pair:
```
A[skeleton_body] = X_anny @ inv(X_skel)
```

Where:
- `X_skel` = skeleton body world pose in neutral
- `X_anny` = ANNY bone world pose in neutral
- `A` = alignment transform (4×4 SE(3) matrix)

## Dependencies

- `numpy`
- `mujoco`
- `torch`
- `anny`
- `trimesh`
- `scipy`
- `loco_mujoco`

## Script Stats

- **Lines of code**: 276
- **Functions**: 5 main functions
- **CLI arguments**: 3 flags
- **Execution time**: ~10-15 seconds (without viz)

## Troubleshooting

### Common Issues

**Q: Script fails with "unknown default class name 'mimic'" error**
A: This has been fixed in the latest version by stripping the `class="mimic"` references from the skeleton XML.

**Q: Script fails with "No such file or directory" for mesh files**
A: This has been fixed by using absolute paths for skeleton mesh files.

**Q: Viewer doesn't open**
A: Make sure `mujoco.viewer` is properly imported. The script now includes this import.

## Testing

To verify the script works correctly:

```bash
# Quick test without visualization
python calibrate_skeleton_to_anny.py --no-viz

# Should output:
# ✓ 14 body pairs mapped
# ✓ RMS rotation error: 0.00 deg
# ✓ RMS translation error: 0.00 cm
# ✓ align_se3.json created (6.8KB)
# ✓ joint_meta.json created (3.9KB)
```
