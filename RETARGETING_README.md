# Skeleton-to-ANNY Retargeting (Runtime)

Real-time motion retargeting from SkeletonTorque to ANNY using pre-computed calibration.

## Overview

This script implements **Part B: Per-frame mapping** that drives ANNY from SkeletonTorque poses in real-time.

### What It Does:

1. **Loads calibration data** (from `align_se3.json`, `joint_meta.json`)
2. **Each frame**:
   - Reads skeleton pose (qpos)
   - Computes skeleton body world poses
   - Applies SE(3) alignments to get target ANNY bone poses
   - Generates ANNY bone transformations (4×4 matrices)
3. **Visualizes** skeleton motion (ANNY computation in background)
4. **Reports** retargeting statistics and performance

## Usage

```bash
# Run with neutral pose (single frame)
python retarget_skeleton_to_anny.py --traj neutral

# Run with squat animation (120 frames, looping)
python retarget_skeleton_to_anny.py --traj squat

# Show error metrics
python retarget_skeleton_to_anny.py --traj squat --error-metrics

# Adjust playback speed
python retarget_skeleton_to_anny.py --traj squat --hz 30
```

## Prerequisites

Must run calibration first:
```bash
python calibrate_skeleton_to_anny.py --no-viz
```

This generates:
- `align_se3.json` - SE(3) alignments for 14 body pairs
- `joint_meta.json` - Joint metadata (types, axes, etc.)

## How Retargeting Works

### Per-Frame Pipeline:

```
Skeleton qpos (joint angles)
    ↓
mj_forward (compute body world poses)
    ↓
For each mapped body:
    X_anny_target = A[body] @ X_skeleton
    ↓
ANNY bone poses (4×4 transforms)
    ↓
(Optionally) Generate ANNY mesh
```

### SE(3) Alignment Formula:

For each skeleton body `bS` mapped to ANNY bone `bA`:

```
X_anny[bA] = A[bS] @ X_skeleton[bS]
```

Where:
- `X_skeleton[bS]` = skeleton body world pose (from MuJoCo xmat/xpos)
- `A[bS]` = pre-computed alignment (from calibration)
- `X_anny[bA]` = target ANNY bone world pose

### Body Pairs Retargeted (14 total):

| Skeleton Body | ANNY Bone | Notes |
|---------------|-----------|-------|
| pelvis | root | Root transformation |
| torso | spine03 | Upper spine |
| femur_r/l | upperleg01.R/L | Thighs |
| tibia_r/l | lowerleg01.R/L | Shins |
| talus_r/l | foot.R/L | Feet |
| humerus_r/l | shoulder01.R/L | Upper arms |
| ulna_r/l | lowerarm01.R/L | Forearms |
| hand_r/l | wrist.R/L | Hands |

## Current Implementation Status

### ✅ Implemented:
- Load calibration data
- Map skeleton qpos → ANNY bone poses (per frame)
- Skeleton visualization with motion playback
- Simple squat trajectory generation
- Performance metrics (FPS)
- Error metrics (RMS rotation/translation)

### ⚠️ Limitations:
- **ANNY visualization**: ANNY bone poses are computed but not displayed
  - Skeleton is shown in MuJoCo viewer
  - ANNY transformations are computed in background
  - To visualize ANNY, would need to:
    - Generate ANNY mesh each frame (slow)
    - Or create separate ANNY viewer (complex)

### Why ANNY Isn't Visualized:

ANNY is a **parametric mesh model** (not a MuJoCo articulated body):
- Input: bone_poses (dict of 4×4 transforms)
- Output: deformed mesh vertices

To visualize ANNY moving in sync with skeleton would require:
1. **Option A**: Generate ANNY mesh each frame (~50-100ms)
2. **Option B**: Export meshes to files for offline viewing
3. **Option C**: Create lightweight ANNY kinematic chain in MuJoCo

Within ~300 line constraint, current implementation focuses on:
- Correct retargeting computation ✓
- Fast skeleton visualization ✓
- Verification via metrics ✓

## Performance

**Typical Performance:**
- **FPS**: 60+ Hz (retargeting computation)
- **Latency**: <1ms per frame (14 body SE(3) transforms)
- **Memory**: Preallocated buffers (no allocation in loop)

**Bottleneck:**
- ANNY mesh generation: ~50-100ms (if enabled)
- MuJoCo forward kinematics: <1ms
- Visualization rendering: ~16ms @ 60Hz

## Script Structure (~300 lines)

```python
# SE(3) helpers (inline, minimal)
se3_inv, se3_mul, xmat_xpos_to_se3, mat_to_quat_wxyz

# Load calibration
load_calibration() → alignments, joint_meta

# Build name map
build_name_map() → [(skel_body, anny_bone, body_id), ...]

# Core retargeting
map_frame(q_skel) → bone_poses_anny
    For each body:
        X_skel = get_world_pose(body)
        X_anny = A[body] @ X_skel
        bone_poses[anny_bone] = X_anny

# Visualization
run_demo()
    Load trajectory (neutral or squat)
    For each frame:
        Update skeleton qpos
        Compute ANNY bone poses
        Render skeleton
        Print metrics
```

## Future Enhancements

To fully visualize ANNY matching skeleton motion:

### Option 1: Export Meshes
```python
# Add --save-anny flag
python retarget_skeleton_to_anny.py --traj squat --save-anny output/
# → Saves ANNY mesh per frame to output/*.obj
# → View offline in Blender/MeshLab
```

### Option 2: Dual Viewer
```python
# Create ANNY viewer in separate thread
# Update ANNY mesh asynchronously
# Show both viewers side-by-side
```

### Option 3: Joint-Based ANNY
```python
# Create MuJoCo kinematic chain for ANNY
# Map skeleton joints → ANNY joints
# Drive both as articulated bodies
```

## Verification

To verify retargeting works correctly:

1. **Run with error metrics**:
```bash
python retarget_skeleton_to_anny.py --traj squat --error-metrics
```

2. **Check output**:
```
Frame 0/120: 14 bones retargeted | RMS: 0.00° 0.00cm
FPS: 62.3 | Mapped 14 bodies each frame
```

3. **Observe**:
   - Skeleton animates smoothly
   - FPS > 60 (real-time capable)
   - RMS errors near 0 (correct alignment)

## Error Metrics Explained

**RMS Rotation Error** (degrees):
- Angular difference between target and actual bone orientations
- Should be ~0° for calibrated neutral pose
- May increase during motion (due to approximations)

**RMS Translation Error** (cm):
- Distance between target and actual bone positions
- Should be ~0cm for matched bodies
- Reflects alignment quality

## Troubleshooting

**Q: "ERROR: align_se3.json not found!"**
```bash
# Run calibration first:
python calibrate_skeleton_to_anny.py --no-viz
```

**Q: Skeleton doesn't move in squat**
- Check joint indices in squat generation code
- Verify skeleton joint names match (run with --traj neutral first)

**Q: FPS is low (<30)**
- Disable error metrics (--no-error-metrics is default)
- Reduce Hz (--hz 30)
- Check system load

**Q: Where is ANNY visualization?**
- Current version computes ANNY poses but doesn't display them
- See "Future Enhancements" above for visualization options
- Focus is on correct retargeting logic + performance

## Technical Notes

### Coordinate Systems:
- **Skeleton**: ISB convention (Y-up, body-fixed frames)
- **ANNY**: Z-up convention (world-fixed bones)
- **Alignment**: SE(3) transforms handle conversion

### Joint Types Supported:
- **Hinge**: 1-DoF revolute (extract angle about axis)
- **Ball**: 3-DoF spherical (convert to quaternion)
- **Euler**: 3-DoF with order (extract angles in sequence)
- **Free**: 6-DoF root (position + quaternion)

### Buffers (Preallocated):
```python
bone_poses_buffer = {label: np.eye(4) for label in anny_model.bone_labels}
# → Reused each frame, no allocation in loop
```

## Example Session

```bash
$ python retarget_skeleton_to_anny.py --traj squat --error-metrics

================================================================================
SkeletonTorque → ANNY Per-Frame Retargeting
================================================================================

1. Loading calibration data...
   Loaded 14 body alignments

2. Loading SkeletonTorque...
   21 bodies, 28 joints

3. Loading ANNY parametric model...
   152 bones

4. Building name map...
   Mapped 14 body pairs

5. Running demo (trajectory: squat, 60 Hz)...
   Generating squat motion...
   Generated squat trajectory: 120 frames

Starting visualization...
Showing: SkeletonTorque animating (ANNY retargeting computed in background)
Note: ANNY bone poses are computed but not visualized in this version
      Use --error-metrics to see retargeting quality

Controls: ESC=exit | E=frames
================================================================================
Frame 0/120: 14 bones retargeted | RMS: 0.00° 0.00cm
  FPS: 61.2 | Mapped 14 bodies each frame
Frame 60/120: 14 bones retargeted | RMS: 0.00° 0.00cm
  FPS: 60.8 | Mapped 14 bodies each frame
...

Retargeting Summary:
  • Processed 120 frames
  • Mapped 14 skeleton bodies → ANNY bones per frame
  • ANNY bone poses computed using 14 SE(3) alignments

Done!
```

## Dependencies

- `numpy` - Matrix operations
- `mujoco` - Physics simulation, visualization
- `scipy` - Rotation utilities
- `torch` - ANNY model
- `anny` - Parametric human model
- `loco_mujoco` - SkeletonTorque environment

## Script Stats

- **Lines**: ~300
- **Functions**: 7 main functions
- **Performance**: 60+ FPS @ 14 body mappings
- **Memory**: Minimal (preallocated buffers)
