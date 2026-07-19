#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Sync the gfx90a fork trunk with upstream (merge-based fork workflow).
#
# Branch roles for this long-lived gfx90a-compatibility fork:
#   main       -- pure upstream mirror. Only ever fast-forwarded from origin/main
#                 (which is kept in sync with upstream). NEVER commit fork changes here.
#   gfx90a     -- the fork TRUNK: upstream + the gfx90a delta. The only branch upstream
#                 is merged into, and where releases are tagged.
#   dev/*      -- ephemeral feature branches off gfx90a, merged back via PR/merge.
#
# RULE: never merge upstream into a feature branch. Sync the trunk here, then rebase
#       or merge your feature branch onto the updated trunk.
#
# This script: fast-forwards `main` from origin/main, then merges `main` into `gfx90a`.
# Run it ON the trunk with a clean working tree. Resolve any conflicts keeping the
# gfx90a logic, then follow the printed post-merge steps.
#
# Usage:  git checkout gfx90a && bash scripts/sync_upstream.sh

set -uo pipefail

TRUNK="${FORK_TRUNK:-gfx90a}"
MIRROR="${UPSTREAM_MIRROR:-main}"

cur="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${cur}" != "${TRUNK}" ]]; then
    echo "[sync] Run this on the '${TRUNK}' trunk (currently on '${cur}'). Aborting." >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[sync] Working tree not clean — commit or stash first. Aborting." >&2
    exit 1
fi

echo "[sync] Fetching origin..."
git fetch origin || { echo "[sync] fetch failed" >&2; exit 1; }

echo "[sync] Fast-forwarding upstream mirror '${MIRROR}' <- origin/${MIRROR}..."
git checkout "${MIRROR}"
if ! git merge --ff-only "origin/${MIRROR}"; then
    echo "[sync] '${MIRROR}' is not a fast-forward of origin/${MIRROR} — it has diverged." >&2
    echo "[sync] The mirror must stay a pure upstream copy; investigate before syncing." >&2
    git checkout "${TRUNK}"
    exit 1
fi

git checkout "${TRUNK}"
echo "[sync] Merging '${MIRROR}' into '${TRUNK}'..."
git merge --no-ff "${MIRROR}" -m "Merge ${MIRROR} into ${TRUNK} (upstream sync)"
rc=$?

echo ""
if [[ "${rc}" -ne 0 ]]; then
    echo "[sync] Merge stopped with conflicts. Resolve them keeping the gfx90a logic"
    echo "       (arch gating, fail-fasts, glc cache modifiers, K=16 paths — see"
    echo "       docs/gfx90a_triage.md), then 'git commit'."
fi
echo "[sync] Post-merge checklist:"
echo "  1. Re-sync build-fly (pytest imports flydsl from build-fly, NOT source):"
echo "       bash scripts/build.sh -jN && pip install -e ."
echo "  2. Re-run the gfx90a gate:  bash scripts/ci_gfx90a.sh   (expect pass/skip only)"
echo "  3. git push origin ${TRUNK}"
exit "${rc}"
