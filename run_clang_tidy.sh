#!/bin/bash
set -euo pipefail

# This script runs clang-tidy with OS-specific arguments.

# Get the path to the directory containing this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Define the base clang-tidy command.
CLANG_TIDY_BIN=$(command -v clang-tidy)
if [ -z "$CLANG_TIDY_BIN" ]; then
  echo "Error: clang-tidy not found in PATH." >&2
  exit 1
fi

# Define the common arguments.
CLANG_TIDY_ARGS=(
  "--checks=readability-static-definition-in-anonymous-namespace,modernize-concat-nested-namespaces"
  "-fix"
)

# Add macOS-specific arguments.
if [ "$(uname)" == "Darwin" ]; then
  echo "Running clang-tidy with macOS toolchain."
  CLANG_TIDY_ARGS+=(
    "--extra-arg=-isysroot"
    "--extra-arg=$(xcrun --show-sdk-path)"
  )
else
  # This branch would be for Linux, Windows, etc.
  echo "Running clang-tidy with default toolchain."
fi

# Pass the compile_commands.json directory.
# This assumes the file is in the project root.
if [ -f "$PROJECT_ROOT/compile_commands.json" ]; then
  CLANG_TIDY_ARGS+=("-p=$PROJECT_ROOT")
fi

# Run clang-tidy on the files passed by pre-commit.
# $@ is the array of all arguments passed to this script (i.e., the filenames).
$CLANG_TIDY_BIN "${CLANG_TIDY_ARGS[@]}" "$@"
