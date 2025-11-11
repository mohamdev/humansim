#!/usr/bin/env python3
"""
Minimal visualization script for matched ANNY and SkeletonTorque models.

This script uses the MatchedSceneBuilder class to create and display
both models side-by-side with matched anatomical landmarks.
"""

import mujoco
import mujoco.viewer
from scene_builder import MatchedSceneBuilder


def main():
    """
    Display matched anatomical frames for ANNY and SkeletonTorque models.
    """
    print("\n" + "=" * 80)
    print("ANNY + SkeletonTorque: Matched Anatomical Frames Visualization")
    print("=" * 80 + "\n")

    # Create scene builder
    builder = MatchedSceneBuilder(
        anny_offset=(-0.5, 0, 1),
        skeleton_offset=(0.5, 0, 0),
        skin_opacity=0.15,
        frame_size=0.3
    )

    # Build the scene
    try:
        model, data = builder.build()
    except Exception as e:
        print(f"\nError building scene: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print viewer controls
    print("\n" + "=" * 80)
    print("Viewer Controls:")
    print("  - Press E to toggle reference frames")
    print("  - Press T to make models transparent")
    print("  - Press H to hide/show the menu")
    print("  - Press TAB to switch cameras")
    print("  - Drag with LEFT mouse to rotate view")
    print("  - Drag with RIGHT mouse to move view")
    print("  - Scroll to zoom")
    print("  - Press ESC or close window to exit")
    print("=" * 80)
    print("\nMatched anatomical landmarks:")
    for landmark in builder.anatomical_matches.keys():
        print(f"  - {landmark}")
    print("\nModels:")
    print(f"  - ANNY (left, x={builder.anny_offset[0]}): Deformable skin with {model.nskin} skin(s)")
    print(f"  - SkeletonTorque (right, x={builder.skeleton_offset[0]}): Rigid body model")
    print("=" * 80 + "\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable frame visualization for sites and bodies
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE | mujoco.mjtFrame.mjFRAME_BODY

        # Make world origin frame larger
        world_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "world_origin")
        if world_site_id >= 0:
            model.site_size[world_site_id][0] = 0.05

        # Set initial camera view
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 5.0
        viewer.cam.lookat[:] = [0, 0, 1]

        print("Viewer launched! Displaying matched anatomical frames...")
        print("Close the viewer window to exit.\n")

        # Main visualization loop
        while viewer.is_running():
            # Step forward the simulation
            mujoco.mj_forward(model, data)
            viewer.sync()

    print("\nCleaning up...")
    builder.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
