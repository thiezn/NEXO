#!/usr/bin/env bash
set -euo pipefail

###############################################
# Configuration
###############################################
BOOK_DIR="$(dirname "$(realpath "$0")")/../docs"
PORT=4001
SERVE_URL="http://localhost:${PORT}"

cd "$BOOK_DIR" || { echo "❌ Failed to cd into ${BOOK_DIR}"; exit 1; }

###############################################
# Helper: Run command quietly unless it fails #
###############################################
run_quiet() {
    local cmd="$*"
    local tmp
    tmp="$(mktemp)"

    echo "▶ Running: $cmd (quiet mode)"
    if ! bash -c "$cmd" >"$tmp" 2>&1; then
        echo
        echo "❌ Command failed: $cmd"
        echo "------ OUTPUT BEGIN ------"
        cat "$tmp"
        echo "------- OUTPUT END -------"
        rm -f "$tmp"
        exit 1
    fi

    rm -f "$tmp"
}

###############################################
# Helpers: get installed + latest versions    #
###############################################
get_installed_version() {
    local bin="$1"
    if command -v "$bin" >/dev/null 2>&1; then
        "$bin" --version | awk '{print $2}'
    else
        echo "none"
    fi
}

get_latest_version() {
    local crate="$1"
    # cargo search output example: mdbook = "0.4.37"
    cargo search "$crate" --limit 1 | sed -E 's/^[^"]+"([^"]+)".*/\1/'
}

update_crate_if_needed() {
    local crate="$1"
    local binary="$2"

    local installed latest
    installed="$(get_installed_version "$binary")"
    latest="$(get_latest_version "$crate")"

    if [ "$installed" != "$latest" ]; then
        echo "📦 Updating $crate ($installed → $latest)…"
        run_quiet "cargo install $crate"
    else
        echo "✔️ $crate is already up to date ($installed)"
    fi
}

###############################################
# Update mdBook + mdbook-mermaid
###############################################
echo "🔄 Checking for updates…"

update_crate_if_needed "mdbook" "mdbook"
update_crate_if_needed "mdbook-mermaid" "mdbook-mermaid"

###############################################
# Update Mermaid bundle
###############################################
echo "🔄 Updating Mermaid bundle via mdbook-mermaid…"
run_quiet "mdbook-mermaid install ."

###############################################
# Build the book
###############################################
echo "📘 Building the book…"
run_quiet "mdbook build"

###############################################
# Serve the book
###############################################
echo "🚀 Serving book on port ${PORT}…"
echo "📎 URL: ${SERVE_URL}"

# Try to open the default browser
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${SERVE_URL}" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
    open "${SERVE_URL}" >/dev/null 2>&1 || true
fi

# mdbook serve stays in the foreground
mdbook serve --port "${PORT}" --open
