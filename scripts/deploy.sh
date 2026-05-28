#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
cd "$ROOT_DIR" || { echo "❌ Failed to cd into ${ROOT_DIR}"; exit 1; }

ALL_TARGETS=(
	"nexo-gateway"
	"nexo-client"
	"game-extractor"
	"epub-extractor"
	"nexo-node"
	"nexo-ai"
)

print_usage() {
	echo "Usage: ./scripts/deploy.sh [target ...]"
	echo
	echo "Deploy all targets when no arguments are provided."
	echo "Valid targets: ${ALL_TARGETS[*]}"
}

is_valid_target() {
	local candidate="$1"
	local known_target
	for known_target in "${ALL_TARGETS[@]}"; do
		if [[ "$known_target" == "$candidate" ]]; then
			return 0
		fi
	done
	return 1
}

binary_for_target() {
	local target="$1"
	case "$target" in
		"nexo-gateway")
			echo "nexo"
			;;
		*)
			echo "$target"
			;;
	esac
}


if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
	print_usage
	exit 0
fi

DEPLOY_TARGETS=()
if [[ "$#" -eq 0 ]]; then
	DEPLOY_TARGETS=("${ALL_TARGETS[@]}")
else
	for requested_target in "$@"; do
		if ! is_valid_target "$requested_target"; then
			echo "❌ Unknown deploy target: $requested_target"
			print_usage
			exit 1
		fi
		DEPLOY_TARGETS+=("$requested_target")
	done
fi

# Clean up build artifacts
echo "Cleaning up previous build artifacts..."
cargo clean

# First test and build the production apps
# cargo test --release
# echo "Tests passed"

echo "Building the production binaries for: ${DEPLOY_TARGETS[*]}"
BUILD_COMMAND=(cargo build --release)
for deploy_target in "${DEPLOY_TARGETS[@]}"; do
	BUILD_COMMAND+=(-p "$deploy_target")
done
"${BUILD_COMMAND[@]}"

for deploy_target in "${DEPLOY_TARGETS[@]}"; do
	binary_name="$(binary_for_target "$deploy_target")"
	sudo cp "./target/release/${binary_name}" /usr/local/bin
done

# Now change the permissions of the binaries so that they can be executed by anyone.
echo "Setting permissions for the binaries..."
for deploy_target in "${DEPLOY_TARGETS[@]}"; do
	binary_name="$(binary_for_target "$deploy_target")"
	sudo chown root:admin "/usr/local/bin/${binary_name}"
done

# Verify the permissions
# ls -ltrah /usr/local/bin/* | grep "\-rwx"
