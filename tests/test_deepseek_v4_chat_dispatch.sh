#!/bin/sh
set -eu

model=$1
if ! nm floyd | grep -q 'deepseek_v4_chat_run'; then
    echo "floyd is missing built-in deepseek_v4_chat_run" >&2
    exit 1
fi
if [ -e deepseek_v4_chat ]; then
    echo "standalone deepseek_v4_chat binary must not be required" >&2
    exit 1
fi

output=$(printf ':exit\n' | env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd --model "$model" --ctx 64 --ngen 1 2>&1)
printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q 'floyd chat \[DeepSeek V4\]'
printf '%s\n' "$output" | grep -q 'DEEPSEEK_V4_BACKEND backend='
printf '%s\n' "$output" | grep -q '›'

draft_output=$(printf ':exit\n' | env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd --model "$model" --ctx 64 --ngen 1 --draft 3 2>&1)
printf '%s\n' "$draft_output"
printf '%s\n' "$draft_output" |
    grep -q 'DEEPSEEK_V4_SPEC disabled=mtp-not-prepared'

for option in temp top-p system; do
    case "$option" in
        temp) value=0.7 ;;
        top-p) value=0.9 ;;
        system) value=test ;;
    esac
    set +e
    rejected=$(./floyd --model "$model" "--$option" "$value" 2>&1)
    status=$?
    set -e
    printf '%s\n' "$rejected"
    [ "$status" -eq 2 ]
    printf '%s\n' "$rejected" |
        grep -q "DeepSeek V4 chat does not support --$option"
done
