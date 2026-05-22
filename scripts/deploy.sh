#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
cd "$ROOT_DIR" || { echo "❌ Failed to cd into ${ROOT_DIR}"; exit 1; }


# First test and build the production apps
# cargo test --release
# echo "Tests passed"

echo "Building the production binaries..."
cargo build --release

sudo cp ./target/release/nexo /usr/local/bin
sudo cp ./target/release/nexo-client /usr/local/bin
sudo cp ./target/release/game-extractor /usr/local/bin
sudo cp ./target/release/epub-extractor /usr/local/bin
sudo cp ./target/release/nexo-node /usr/local/bin
sudo cp ./target/release/nexo-ai /usr/local/bin

# Now change the permissions of the binaries so that they can be executed by anyone.
echo "Setting permissions for the binaries..."
sudo chown root:admin /usr/local/bin/nexo
sudo chown root:admin /usr/local/bin/nexo-client
sudo chown root:admin /usr/local/bin/game-extractor
sudo chown root:admin /usr/local/bin/epub-extractor
sudo chown root:admin /usr/local/bin/nexo-node
sudo chown root:admin /usr/local/bin/nexo-ai

# Updating local inference tools
source /Users/Mathijs.Mortimer/Development/utilities/.venv/bin/activate
uv pip install --upgrade mlx-vlm mlx-audio

# Verify the permissions
# ls -ltrah /usr/local/bin/* | grep "\-rwx"
