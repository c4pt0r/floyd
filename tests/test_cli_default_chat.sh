#!/bin/sh
set -eu

model=$1
if make -n METAL=0 floyd >/dev/null 2>&1; then
    echo "METAL=0 build unexpectedly succeeded" >&2
    exit 1
fi
help=$(./floyd help)
printf '%s\n' "$help" | grep -q '^usage:'
if printf '%s\n' "$help" | grep -q '^uso:'; then
    echo "help output is not English" >&2
    exit 1
fi
set +e
output=$(printf 'hello\n:exit\n' | env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd --model "$model" --ngen 1 2>&1)
status=$?
set -e
printf '%s\n' "$output"
[ "$status" -eq 0 ]
printf '%s\n' "$output" | grep -q "floyd chat \[Moonlight\]"
printf '%s\n' "$output" | grep -q "MOONLIGHT_BACKEND backend=metal-moonlight"
if printf '%s\n' "$output" | grep -q "Motore C GLM"; then
    echo "Moonlight chat unexpectedly used the legacy CPU runtime" >&2
    exit 1
fi
printf '%s\n' "$output" | grep -q "›"
printf '%s\n' "$output" | grep -q "◆"

set +e
too_long=$(./floyd run --model "$model" --ctx 8 --ngen 1 \
    --prompt "This prompt is intentionally too long for an eight token context." \
    --temp 0 2>&1)
too_long_status=$?
set -e
printf '%s\n' "$too_long"
[ "$too_long_status" -ne 0 ]
printf '%s\n' "$too_long" | grep -Eq 'too long|context'
