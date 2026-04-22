#!/bin/bash

# Launch G1 Upper Body Scene for Stage 1

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

echo "=========================================="
echo "Stage 1: G1 Upper Body End-Effector Tracking"
echo "=========================================="

# Run the stage 1 script
cd "$PROJECT_DIR"
python scripts/run_stage1.py --duration 10.0 --save-video

echo ""
echo "To visualize the recorded data, run:"
echo "  python scripts/visualize_data.py --latest"