#!/bin/sh
set -eu

failed=0
legacy_files=$(git ls-files | grep -E '(^|/)(v4_|test_v4_|make_v4_)' || true)
if [ -n "$legacy_files" ]; then
    echo "legacy DeepSeek V4 filenames:" >&2
    printf '%s\n' "$legacy_files" >&2
    failed=1
fi

legacy_c=$(rg -n '\bv4_|\bV4[A-Z_]' --glob='*.c' --glob='*.h' || true)
if [ -n "$legacy_c" ]; then
    echo "legacy DeepSeek V4 C symbols:" >&2
    printf '%s\n' "$legacy_c" >&2
    failed=1
fi

legacy_targets=$(rg -n 'test-v4-' Makefile || true)
if [ -n "$legacy_targets" ]; then
    echo "legacy DeepSeek V4 Make targets:" >&2
    printf '%s\n' "$legacy_targets" >&2
    failed=1
fi

[ "$failed" -eq 0 ]
echo "DeepSeek V4 naming: ok"
