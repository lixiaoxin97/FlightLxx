#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/../gazebo/plugins"
BUILD_DIR="$SCRIPT_DIR/../build/state_injector"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_DIR" --parallel "$(nproc)"
printf 'Plugin: %s\nCLI:    %s\n' \
  "$BUILD_DIR/libflightlxx_state_injector.so" \
  "$BUILD_DIR/flightlxx_state_injector_cli"
