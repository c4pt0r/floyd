#!/bin/sh
set -eu

model=$1
trace=$(mktemp /tmp/floyd-deepseek-v4-spec.XXXXXX)
trap 'rm -f "$trace"' EXIT

greedy=$(PROMPT=hello NGEN=2 DSPARK_SPEC=0 ./deepseek_v4_chat "$model" 64 2)
spec=$(PROMPT=hello NGEN=2 DSPARK_SPEC=1 DEEPSEEK_V4_CHAT_TRACE=1 \
    ./deepseek_v4_chat "$model" 64 2 2>"$trace")
printf 'greedy: %s\nspec:   %s\n' "$greedy" "$spec"
test "$greedy" = "$spec"
grep -q '^DEEPSEEK_V4_SPEC ' "$trace"
cat "$trace"
