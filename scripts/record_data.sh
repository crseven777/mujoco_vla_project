#!/bin/bash

# Stage 1 data recording entrypoint.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

DURATION="${1:-10.0}"
OUTPUT_DIR="${2:-data/raw}"

cd "$PROJECT_DIR"
python scripts/run_stage1.py --duration "$DURATION" --output-dir "$OUTPUT_DIR" --save-video
