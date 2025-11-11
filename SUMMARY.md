# Calibration Script - Implementation Summary

## ✅ What I Implemented (Option A)

You asked for **Option A: Apply the root alignment to move the entire skeleton to overlay ANNY**

### Changes Made:

1. **Modified `create_combined_scene()` function**:
   - Now takes `alignments` parameter
   - Extracts root (pelvis) alignment transformation
   - Wraps skeleton in a parent body with position and rotation from alignment

2. **Added alignment application**:
   ```xml
   <body name="skel_aligned" pos="[0.000, 0.945, -0.238]" quat="[...]">
       <!-- skeleton body here -->
   </body>
   ```

3. **Updated visualization messages**:
   - Clarified that skeleton is aligned to overlay ANNY
   - Added diagnostic output showing root alignment values

## What the Alignment Does

### Root Transformation Applied:
```
Translation: [0.0, 0.945, -0.238] meters
Rotation:    75.84° around X-axis
```

### What This Means:
- **Translation**: Moves skeleton up by ~0.95m (ANNY's root is higher)
- **Rotation**: Converts ISB coordinate frame → ANNY coordinate frame
  - ISB: Y-axis points up (longitudinal)
  - ANNY: Z-axis points up (vertical)
  - ~76° rotation converts between these conventions

### Visual Result:
✓ Both models now **overlap** at the same pelvis/root location
✓ Skeleton is positioned "inside" the ANNY mesh
✓ Models share the same spatial reference frame

## Why They Don't Match Perfectly

Even with alignment applied, you'll see some differences:

1. **Different proportions**:
   - Skeleton: Biomechanical model (ISB standard)
   - ANNY: Parametric mesh (generic human)

2. **Different detail levels**:
   - Skeleton: 21 bodies, 14 matched to ANNY
   - ANNY: 152 bones total

3. **Only root alignment**:
   - We apply ONE transformation (pelvis alignment) to entire skeleton
   - This is a **rigid body transformation** (moves whole skeleton as one piece)
   - Individual limbs may not perfectly match ANNY's bone positions

4. **Different neutral poses**:
   - Both models are in "neutral" pose but may define it slightly differently
   - Skeleton: Arms at sides
   - ANNY: T-pose or A-pose (depending on model version)

## What Each Alignment in align_se3.json Is For

The script computes **14 body alignments**, but only uses the pelvis one for visualization:

| Body Pair | Alignment Purpose |
|-----------|------------------|
| pelvis → root | **USED for visualization** (moves entire skeleton) |
| torso → spine03 | For retargeting spine orientation |
| femur_r/l → upperleg01.R/L | For retargeting hip/thigh motion |
| tibia_r/l → lowerleg01.R/L | For retargeting knee/shin motion |
| talus_r/l → foot.R/L | For retargeting ankle/foot motion |
| humerus_r/l → shoulder01.R/L | For retargeting shoulder/arm motion |
| ulna_r/l → lowerarm01.R/L | For retargeting elbow/forearm motion |
| hand_r/l → wrist.R/L | For retargeting wrist/hand motion |

## Next Steps: Motion Retargeting

To actually use all 14 alignments for motion transfer:

1. **Capture skeleton motion** (qpos over time)
2. **For each frame**:
   ```python
   for skel_body, anny_bone in body_pairs:
       # Get skeleton body world pose
       X_skel = get_world_pose(skel_body)

       # Apply alignment to get target ANNY pose
       A = alignments[skel_body]
       X_anny_target = A @ X_skel

       # Convert to ANNY bone parameters
       anny_bone_pose[anny_bone] = X_anny_target
   ```
3. **Generate ANNY mesh** with retargeted bone poses
4. **Render animation**

The calibration provides the **A** matrices needed for step 2.

## Files Created

1. `calibrate_skeleton_to_anny.py` (283 lines) - Main script with Option A implemented
2. `align_se3.json` - 14 body pair alignments (SE(3) transforms)
3. `joint_meta.json` - 28 skeleton joint metadata
4. `ALIGNMENT_EXPLAINED.md` - Detailed explanation of alignment
5. `check_alignment.py` - Quick script to inspect alignment values
6. `CALIBRATION_README.md` - User documentation
7. `test_calibration.sh` - Automated test suite

## Usage

```bash
# Run with visualization (models overlaid)
python calibrate_skeleton_to_anny.py

# Check what alignment is being applied
python check_alignment.py

# Run tests
./test_calibration.sh
```

## Expected Visualization

When you run the script with visualization:

1. **ANNY mesh** (transparent green): Human mesh at origin
2. **Skeleton** (opaque bones): Overlaid on ANNY, aligned via root transform
3. **Both centered** at the same pelvis/root location
4. **Coordinate frames** (press E): Show body orientations

The models should look like:
```
    ┌─────────────┐
    │   ANNY mesh │ (transparent)
    │   (outside) │
    │     ┌───┐   │
    │     │ ☐ │   │ <- Skeleton bones
    │     └───┘   │    (inside, overlapping)
    └─────────────┘
```

## Troubleshooting

**Q: Models still look separate?**
- Make sure you're running the latest version with Option A implemented
- Check `check_alignment.py` output - translation should be non-zero
- Press T to adjust transparency and see overlap

**Q: Models match perfectly at pelvis but not limbs?**
- This is expected! Only root alignment is applied (rigid body transform)
- Limbs have different proportions between skeleton and ANNY
- For perfect limb matching, you'd need full retargeting (using all 14 alignments)

**Q: What's the RMS error of 0.00?**
- This just verifies the calibration math is correct
- It doesn't mean visual alignment is perfect
- See ALIGNMENT_EXPLAINED.md for details
