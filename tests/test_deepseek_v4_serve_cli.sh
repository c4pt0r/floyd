#!/bin/sh
set -eu

help=$(./floyd help)
printf '%s\n' "$help" | grep -q 'serve.*--stdio'
printf '%s\n' "$help" | grep -q -- '--prefix-cache-mb'
printf '%s\n' "$help" | grep -q -- '--ds4-model'
printf '%s\n' "$help" | grep -q -- '--ds4-support'

model=$(mktemp -d /tmp/floyd-dsv4-serve-cli.XXXXXX)
trap 'rm -rf "$model"' EXIT INT TERM
printf '{"model_type":"deepseek_v4"}\n' >"$model/config.json"

set +e
missing_stdio=$(env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd serve --model "$model" --prefix-cache-mb 0 2>&1)
missing_status=$?
set -e
[ "$missing_status" -eq 2 ]
printf '%s\n' "$missing_stdio" | grep -q 'serve requires --stdio'

set +e
bad_budget=$(env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd serve --model "$model" --stdio --prefix-cache-mb -1 2>&1)
bad_status=$?
set -e
[ "$bad_status" -eq 2 ]
printf '%s\n' "$bad_budget" | grep -q 'prefix-cache-mb'

set +e
dispatch=$(env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd serve --model "$model" --stdio --ctx 512 --ngen 1 --draft 2 \
    --prefix-cache-mb 0 --ds4-model /missing/base.gguf \
    --ds4-support /missing/support.gguf --trace 2>&1)
dispatch_status=$?
set -e
[ "$dispatch_status" -eq 2 ]
printf '%s\n' "$dispatch" | grep -Eq \
    'FLOYD_DEEPSEEK_V4_DS4_GGUF is not a file|requires.*METAL'

printf 'DeepSeek V4 serve CLI tests: ok\n'
