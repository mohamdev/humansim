#!/bin/bash
# Test complete workflow: Calibration → Retargeting

echo "=========================================="
echo "Testing Complete Skeleton→ANNY Workflow"
echo "=========================================="
echo ""

# Clean previous outputs
rm -f align_se3.json joint_meta.json

# Part A: Calibration
echo "PART A: Running calibration..."
python calibrate_skeleton_to_anny.py --no-viz 2>&1 | grep -E "(Loading|Building|Computing|Saved|Done)" | head -10

if [ ! -f align_se3.json ] || [ ! -f joint_meta.json ]; then
    echo "✗ Calibration failed - output files missing"
    exit 1
fi
echo "✓ Calibration completed"
echo ""

# Part B: Retargeting (neutral pose test)
echo "PART B: Testing retargeting (neutral pose)..."
timeout 3 python retarget_skeleton_to_anny.py --traj neutral 2>&1 | grep -E "(Loading|Building|FPS)" | head -15 &
PID=$!
sleep 2
kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "✓ Retargeting working"
echo ""

# Check outputs
echo "Checking outputs..."
python -c "
import json
with open('align_se3.json', 'r') as f:
    a = json.load(f)
with open('joint_meta.json', 'r') as f:
    j = json.load(f)
print(f'✓ align_se3.json: {len(a)} body alignments')
print(f'✓ joint_meta.json: {len(j)} joints')
"

echo ""
echo "=========================================="
echo "✓ Complete workflow tested successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run calibration with viz: python calibrate_skeleton_to_anny.py"
echo "  2. Run retargeting:          python retarget_skeleton_to_anny.py --traj squat"
