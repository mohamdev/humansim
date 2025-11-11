#!/usr/bin/env python3
"""
Quick script to visualize what the alignment transform does.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Load alignment
with open('align_se3.json', 'r') as f:
    alignments = json.load(f)

root_align = alignments['pelvis']
t = np.array(root_align['translation'])
R = np.array(root_align['rotation'])

print("="*80)
print("Root (Pelvis) Alignment: Skeleton → ANNY")
print("="*80)
print()
print("This transformation is applied to the skeleton's root to align it with ANNY.")
print()

print("Translation (meters):")
print(f"  X: {t[0]:+.4f} m  (left/right)")
print(f"  Y: {t[1]:+.4f} m  (forward/back - ANNY coordinate)")
print(f"  Z: {t[2]:+.4f} m  (up/down - ANNY coordinate)")
print()

euler = Rot.from_matrix(R).as_euler('xyz', degrees=True)
print("Rotation:")
print(f"  Around X: {euler[0]:+.2f}°  (pitch)")
print(f"  Around Y: {euler[1]:+.2f}°  (yaw)")
print(f"  Around Z: {euler[2]:+.2f}°  (roll)")
print()

print("What this means:")
print("  • The skeleton is moved UP by ~{:.2f}m and BACK by ~{:.2f}m".format(t[1], abs(t[2])))
print("  • The skeleton is rotated ~{:.1f}° around X-axis".format(euler[0]))
print("  • This rotation converts ISB coordinate frame → ANNY coordinate frame")
print()

print("Coordinate System Conversion (ISB → ANNY):")
print("  ISB:  X=antero-posterior, Y=longitudinal(up), Z=medio-lateral")
print("  ANNY: X=medio-lateral,   Y=postero-anterior, Z=upward")
print("  The ~76° rotation converts Y-up (ISB) to Z-up (ANNY)")
print()

print("In the visualization:")
print("  ✓ The skeleton is wrapped in a body with pos=[{:.3f}, {:.3f}, {:.3f}]".format(t[0], t[1], t[2]))
print("  ✓ And rotation = {:.1f}° around X-axis".format(euler[0]))
print("  ✓ This moves the skeleton to overlay the ANNY mesh")
print()

print("="*80)
print("Expected Result in Viewer:")
print("="*80)
print("  • Both models should be centered at the same pelvis/root location")
print("  • The skeleton should be 'inside' the ANNY mesh (overlapping)")
print("  • They won't match perfectly due to different proportions")
print("  • Press E to see body frames, T to adjust transparency")
print()
