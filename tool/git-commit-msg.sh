#!/bin/sh

COMMIT_MESSAGE=$(cat $1)
COMMIT_FORMAT_REGEX="^(feat|fix|perf|refactor|build|style|test|ci|doc|chore)(\((.*)\))?:( #([0-9]+))? (.*)$"

if ! [[ "$COMMIT_MESSAGE" =~ $COMMIT_FORMAT_REGEX ]]; then
  cat <<\EOF
Error: This commit message is rejected.
EOF
  exit 1
fi
