#!/bin/sh
set -eu

model=$1
trace=$(mktemp /tmp/floyd-deepseek-v4-spec.XXXXXX)
trap 'rm -f "$trace"' EXIT

greedy=$(SNAP="$model" CHAT=1 PROMPT=hello CTX=64 NGEN=2 DSPARK_SPEC=0 ./floyd)
spec=$(SNAP="$model" CHAT=1 PROMPT=hello CTX=64 NGEN=2 DSPARK_SPEC=1 \
    DEEPSEEK_V4_CHAT_TRACE=1 ./floyd 2>"$trace")
printf 'greedy: %s\nspec:   %s\n' "$greedy" "$spec"
test "$greedy" = "$spec"
grep -q '^DEEPSEEK_V4_SPEC ' "$trace"
cat "$trace"
