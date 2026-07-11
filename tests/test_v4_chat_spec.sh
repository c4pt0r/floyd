#!/bin/sh
set -eu

model=$1
trace=$(mktemp /tmp/floyd-v4-spec.XXXXXX)
trap 'rm -f "$trace"' EXIT

greedy=$(PROMPT=hello NGEN=2 DSPARK_SPEC=0 ./v4_chat "$model" 64 2)
spec=$(PROMPT=hello NGEN=2 DSPARK_SPEC=1 V4_CHAT_TRACE=1 \
    ./v4_chat "$model" 64 2 2>"$trace")
printf 'greedy: %s\nspec:   %s\n' "$greedy" "$spec"
test "$greedy" = "$spec"
grep -q '^V4_SPEC ' "$trace"
cat "$trace"
