#!/usr/bin/env bash
# Copy persisted_assets/ to a fast scratch location and symlink assets -> there.
#
# The repo keeps two views of the model data:
#   - persisted_assets/  (HOME, slow but persistent)        ← source of truth
#   - assets/            (symlink to netscratch, ephemeral) ← what the code reads
#
# All code references "assets/llama3/..." relative paths, so after running
# this script those reads happen on netscratch instead of HOME.
#
# Usage:
#   tools/sync_to_netscratch.sh <target_path>
#
# Example:
#   tools/sync_to_netscratch.sh /n/netscratch/kozinsky_lab/Lab/okulseng/cs265-assets
#
# Idempotent: re-running just rsyncs again (so persisted_assets edits propagate).
# rsync supports resume on interruption.

set -euo pipefail


TARGET=/n/netscratch/kozinsky_lab/Lab/okulseng/cs265-assets
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/persisted_assets"
LINK="$REPO_ROOT/assets"

if [[ ! -d "$SRC" ]]; then
    echo "Error: $SRC does not exist." >&2
    exit 1
fi

# Refuse to clobber a real directory at $LINK; replacing a symlink is fine.
if [[ -e "$LINK" && ! -L "$LINK" ]]; then
    echo "Error: $LINK exists and is not a symlink. Refusing to clobber." >&2
    echo "Move it aside or remove it manually first." >&2
    exit 1
fi

mkdir -p "$TARGET"

# Guard against pointing at $SRC itself.
if [[ "$(realpath "$TARGET")" == "$(realpath "$SRC")" ]]; then
    echo "Error: target equals source ($SRC). That would be a self-copy." >&2
    exit 1
fi

echo "Syncing $SRC/  ->  $TARGET/"
rsync -aP "$SRC/" "$TARGET/"

# ln -sfn atomically replaces an existing symlink (and won't follow it into
# the target directory the way plain `ln -sf` would).
ln -sfn "$TARGET" "$LINK"

echo
echo "Done."
echo "  $LINK -> $(readlink "$LINK")"
