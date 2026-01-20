#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_URL="git@github.com:sigurdurhaukur/OCR-icelandic.git"
REMOTE_ID="upstream"
UPSTREAM_BRANCH="main"
LOCAL_BRANCH="main"
TEMP_BRANCH="upstream_temp"

# Add upstream remote if missing
if ! git remote | grep -q "^${REMOTE_ID}$"; then
    echo "Adding upstream remote..."
    git remote add "$REMOTE_ID" "$UPSTREAM_URL"
fi

echo "Fetching upstream remote..."
git fetch "$REMOTE_ID"

echo "Creating branch '${TEMP_BRANCH}' from ${REMOTE_ID}/${UPSTREAM_BRANCH}..."
git branch -f "$TEMP_BRANCH" "$REMOTE_ID/$UPSTREAM_BRANCH"

echo "Checking out local ${LOCAL_BRANCH}..."
git checkout "$LOCAL_BRANCH"

echo "Merging ${TEMP_BRANCH} into ${LOCAL_BRANCH}..."
git merge "$TEMP_BRANCH"

echo "Done."
