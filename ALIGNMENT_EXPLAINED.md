# What Does the Calibration Alignment Do?

## Overview

The calibration script computes **SE(3) transformation matrices** that align the SkeletonTorque biomechanical model to the ANNY parametric mesh model.

## What Changed (Option A Implementation)

### Before (Original Code)
- ANNY mesh displayed at position `(0, 0, 0)` in neutral pose
- Skeleton displayed at its own position (separate from ANNY)
- **Models were NOT overlapping** - just shown side by side

### After (With Root Alignment Applied)
- ANNY mesh still at position `(0, 0, 0)` in neutral pose
- **Skeleton is now positioned/rotated using the root (pelvis) alignment transformation**
- The skeleton's root is moved to overlay ANNY's root
- **Models should now be overlapping in the same space**

## How It Works

### 1. Compute Alignment (Step 4)
For each body pair, compute: **`A[skeleton_body] = X_anny @ inv(X_skel)`**

For the pelvis (root):
```
A_root = X_anny_root @ inv(X_skel_pelvis)
```

This transformation tells us: "To move skeleton's pelvis frame to ANNY's root frame, apply A_root"

### 2. Apply Alignment (Visualization)
The skeleton body is wrapped in a parent body with transformation:
```xml
<body name="skel_aligned" pos="[tx, ty, tz]" quat="[qw, qx, qy, qz]">
    <!-- skeleton body here -->
</body>
```

Where:
- `pos` = translation from `A_root`
- `quat` = rotation from `A_root` (converted to quaternion)

### 3. Result
The entire skeleton is rigidly transformed to align its pelvis with ANNY's root bone.

## Expected Visualization

When you run `python calibrate_skeleton_to_anny.py`, you should see:

1. **ANNY mesh** (transparent, greenish): The human mesh at origin
2. **Skeleton** (opaque, bone meshes): Overlaid on top of ANNY
3. Both models centered around the **same pelvis/root location**
4. The skeleton should be "inside" or closely matching ANNY's body structure

### Tips for Viewing:
- Press **E** to toggle reference frames (helps see bone orientations)
- Press **T** to toggle transparency (make skeleton more transparent)
- **Rotate the view** to see the overlay from different angles
- The models may not perfectly match because:
  - Different anatomical proportions (skeleton vs ANNY mesh)
  - Skeleton has fewer bodies (14 matched) vs ANNY (152 bones)
  - Only root alignment is applied (not per-body alignment)

## What About the Other Body Alignments?

The script computes alignments for **14 body pairs**:
- pelvis → root (USED for visualization)
- torso → spine03
- femur_r/l → upperleg01.R/L
- tibia_r/l → lowerleg01.R/L
- talus_r/l → foot.R/L
- humerus_r/l → shoulder01.R/L
- ulna_r/l → lowerarm01.R/L
- hand_r/l → wrist.R/L

Currently, **only the root (pelvis) alignment is applied** to move the entire skeleton rigidly.

### Why Not Apply All Alignments?

The skeleton is a **rigid kinematic chain**. If we applied per-body alignments:
- It would **break the skeleton** (bodies would disconnect)
- We'd need to solve inverse kinematics to find joint angles
- This would be **retargeting**, not calibration

## Use Cases

### What This Calibration IS For:
✓ **Motion retargeting**: Convert skeleton joint angles → ANNY bone poses
✓ **Coordinate system conversion**: Map between ISB and ANNY conventions
✓ **Neutral pose alignment**: Verify the models align in rest pose

### What This Calibration is NOT:
✗ **Perfect visual overlay**: The models have different proportions
✗ **Per-body fitting**: Only root alignment is applied for visualization
✗ **Skinning/deformation**: ANNY mesh doesn't deform to match skeleton

## Next Steps for Motion Retargeting

To use these calibration constants for actual motion retargeting:

1. **Capture skeleton motion** (joint angles over time)
2. **For each frame**:
   - Apply root alignment to get ANNY root pose
   - For each other body, use its alignment + skeleton's world pose
   - Convert to ANNY bone transformations
3. **Generate ANNY mesh** with retargeted bone poses
4. **Render animation**

The calibration constants in `align_se3.json` are the **A[skeleton_body]** transforms needed for step 2.

## Alignment Values

Check the root alignment being applied:
```bash
python -c "import json; a=json.load(open('align_se3.json')); print('Root translation:', a['pelvis']['translation']); print('Root rotation (3x3):'); import numpy as np; print(np.array(a['pelvis']['rotation']))"
```

Typical values:
- Translation: `[~0.0, ~0.9, ~-0.2]` (ANNY root is higher in Z)
- Rotation: Mostly identity with some rotation around X-axis (coordinate frame difference)
