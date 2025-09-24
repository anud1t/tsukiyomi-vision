#!/usr/bin/env bash
set -euo pipefail

# Author: anudit
# Purpose: Install moondream-station (PyPI or source) after venv is activated.

if ! command -v python &>/dev/null; then
  echo "python not found. Activate your virtual environment first (source venv/bin/activate)." >&2
  exit 1
fi

echo "Installing moondream-station from PyPI..."
pip install --upgrade pip
pip install --upgrade moondream-station

echo "Optionally install from source (uncomment to use):"
echo "  git clone https://github.com/m87-labs/moondream-station.git"
echo "  cd moondream-station && pip install -e ."

echo "Verifying installation..."
if ! command -v moondream-station &>/dev/null; then
  echo "moondream-station CLI not found on PATH. Ensure venv/bin is in PATH and reinstall if needed." >&2
  exit 1
fi

echo "Installation complete. You can launch with: moondream-station"

