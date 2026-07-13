#!/bin/sh
set -eu

model=$1
trace=$(mktemp /tmp/floyd-deepseek-v4-spec.XXXXXX)
trap 'rm -f "$trace"' EXIT

greedy=$(./floyd run --model "$model" --prompt hello --ctx 64 --ngen 2 \
    --draft 1)
spec=$(./floyd run --model "$model" --prompt hello --ctx 64 --ngen 2 \
    --draft 3 --trace 2>"$trace")
printf 'greedy: %s\nspec:   %s\n' "$greedy" "$spec"
test "$greedy" = "$spec"
grep -q '^DEEPSEEK_V4_SPEC ' "$trace"
cat "$trace"
