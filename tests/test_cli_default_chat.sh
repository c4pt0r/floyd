#!/bin/sh
set -eu

model=$1
set +e
output=$(printf 'hello\n:exit\n' | env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd --model "$model" --no-kvsave --ngen 1 2>&1)
status=$?
set -e
printf '%s\n' "$output"
[ "$status" -eq 0 ]
printf '%s\n' "$output" | grep -q "floyd chat \[Moonlight\]"
printf '%s\n' "$output" | grep -q "›"
printf '%s\n' "$output" | grep -q "◆"
