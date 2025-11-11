#!/bin/bash
# Quick test script for calibration

echo "=========================================="
echo "Testing Calibration Script"
echo "=========================================="
echo ""

# Test 1: List bodies
echo "1. Testing --list-bodies..."
python calibrate_skeleton_to_anny.py --list-bodies 2>&1 | grep -q "pelvis.*→.*root"
if [ $? -eq 0 ]; then
    echo "   ✓ Body mapping works"
else
    echo "   ✗ Body mapping failed"
    exit 1
fi

# Test 2: Run calibration without viz
echo ""
echo "2. Testing calibration (no viz)..."
python calibrate_skeleton_to_anny.py --no-viz 2>&1 | grep -q "Done! Calibration complete"
if [ $? -eq 0 ]; then
    echo "   ✓ Calibration completed"
else
    echo "   ✗ Calibration failed"
    exit 1
fi

# Test 3: Check output files
echo ""
echo "3. Checking output files..."
if [ -f align_se3.json ] && [ -f joint_meta.json ]; then
    SIZE_ALIGN=$(stat -c%s align_se3.json 2>/dev/null || stat -f%z align_se3.json 2>/dev/null)
    SIZE_JOINT=$(stat -c%s joint_meta.json 2>/dev/null || stat -f%z joint_meta.json 2>/dev/null)

    if [ "$SIZE_ALIGN" -gt 1000 ] && [ "$SIZE_JOINT" -gt 1000 ]; then
        echo "   ✓ Output files created (align_se3.json: ${SIZE_ALIGN} bytes, joint_meta.json: ${SIZE_JOINT} bytes)"
    else
        echo "   ✗ Output files too small"
        exit 1
    fi
else
    echo "   ✗ Output files not found"
    exit 1
fi

# Test 4: Validate JSON
echo ""
echo "4. Validating JSON structure..."
python -c "import json; a=json.load(open('align_se3.json')); j=json.load(open('joint_meta.json')); assert len(a)==14; assert len(j)==28; print('   ✓ JSON valid: 14 alignments, 28 joints')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ✗ JSON validation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python calibrate_skeleton_to_anny.py        # With visualization"
echo "  python calibrate_skeleton_to_anny.py --no-viz  # Without visualization"
