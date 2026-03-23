#!/usr/bin/env bash

# https://corrode.dev/blog/tips-for-faster-rust-compile-times/#find-the-slow-crate-in-your-codebase

set -euo pipefail

# echo "Running cargo machette to remove any unused dependencies... Note that it could have false positives."
# cargo install cargo-machete
# cargo machete --with-metadata

echo "Updating Rust toolchain to the latest stable version..."
rustup update stable

echo "Running cargo clean to remove old build artifacts..."
cargo clean

echo "Updating Cargo dependencies..."
cargo update

# This needs ```cargo install cargo-edit``` first
echo "Bumping all dependencies to their latest versions..."
cargo upgrade --recursive --verbose

# echo "Disable unused features in dependencies... (DISABLED FOR NOW, IT TAKES A LOT OF TIME AND SPACE)"
# https://github.com/ToBinio/cargo-features-manager
# install with cargo install cargo-features-manager
# cargo features prune

# echo "Checking for outdated dependencies..."
# OUTDATED=$(cargo outdated --depth=1 --exit-code 0 || true)
# if [[ -n "$OUTDATED" ]]; then
#     echo "The following dependencies are outdated:"
#     echo "$OUTDATED"
#     echo "Please review and update them as necessary."
# else
#     echo "All dependencies are up to date."
# fi

# echo "Checking tree for duplicate dependencies..."
# cargo tree -d || echo "No duplicate dependencies found."

# echo "Checking timings for longest build times"
# cargo build --timings=json # only on nightly
# cargo build --timings -vv

# echo "Check what takes most space in executables"
# cargo install cargo-bloat --no-default-features
# cargo bloat --release --crates --time
