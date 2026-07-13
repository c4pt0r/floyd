#!/bin/sh
set -eu

model=$1
output=$(mktemp -t floyd-deepseek-v4-serve.XXXXXX)
diagnostics=$(mktemp -t floyd-deepseek-v4-serve-stderr.XXXXXX)
trap 'rm -f "$output" "$diagnostics"' EXIT INT TERM

printf '%s\n%s\n' \
    '{"id":"first","system":"Answer with one short word.","prompt":"Say alpha.","max_tokens":1,"draft":2}' \
    '{"id":"second","system":"Answer with one short word.","prompt":"Say beta.","max_tokens":1,"draft":2}' |
    ./floyd serve --stdio --model "$model" --ctx 512 --ngen 1 --draft 3 \
        --prefix-cache-mb 256 >"$output" 2>"$diagnostics"

[ "$(wc -l <"$output" | tr -d ' ')" -eq 2 ]
first=$(sed -n '1p' "$output")
second=$(sed -n '2p' "$output")
printf '%s\n%s\n' "$first" "$second"
printf '%s\n' "$first" | grep -q '"id":"first"'
printf '%s\n' "$first" | grep -q '"cache_hit":false'
printf '%s\n' "$first" | grep -q '"error":null'
printf '%s\n' "$second" | grep -q '"id":"second"'
printf '%s\n' "$second" | grep -q '"cache_hit":true'
printf '%s\n' "$second" | grep -q '"error":null'
prefix=$(printf '%s\n' "$second" |
    sed -n 's/.*"cache_prefix_tokens":\([0-9][0-9]*\).*/\1/p')
[ -n "$prefix" ]
[ "$prefix" -gt 0 ]
grep -q 'DEEPSEEK_V4_SERVE transport=stdio backend=metal-ds4' "$diagnostics"
printf 'DeepSeek V4 serve official: cache_prefix_tokens=%s\n' "$prefix"
