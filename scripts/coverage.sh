#!/usr/bin/env bash
# Generate test coverage report using cargo-llvm-cov.
#
# Usage:
#   ./coverage.sh          # Print summary to terminal
#   ./coverage.sh --html   # Generate HTML report in target/coverage/html/
#
# Requires: cargo-llvm-cov (cargo install cargo-llvm-cov) + llvm-tools-preview

set -euo pipefail

if [ "${1:-}" = "--html" ]; then
    cargo llvm-cov --workspace --html --no-cfg-coverage --output-dir target/coverage
    echo "Coverage report: target/coverage/html/index.html"
else
    cargo llvm-cov --workspace --no-cfg-coverage --skip-functions
fi
