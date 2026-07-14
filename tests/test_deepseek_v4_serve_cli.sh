#!/bin/sh
set -eu

help=$(./floyd help)
printf '%s\n' "$help" | grep -q 'serve.*--model'
printf '%s\n' "$help" | grep -q -- '--stdio'
printf '%s\n' "$help" | grep -q -- '--host'
printf '%s\n' "$help" | grep -q -- '--port'
printf '%s\n' "$help" | grep -q -- '--api-key'
printf '%s\n' "$help" | grep -q -- '--served-model-name'
printf '%s\n' "$help" | grep -q -- '--prefix-cache-mb'
printf '%s\n' "$help" | grep -q -- '--ds4-model'
printf '%s\n' "$help" | grep -q -- '--ds4-support'

model=$(mktemp -d /tmp/floyd-dsv4-serve-cli.XXXXXX)
trap 'rm -rf "$model"' EXIT INT TERM
printf '{"model_type":"deepseek_v4"}\n' >"$model/config.json"

set +e
http_defaults=$(env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd serve --model "$model" --prefix-cache-mb 0 2>&1)
http_status=$?
set -e
[ "$http_status" -eq 2 ]
printf '%s\n' "$http_defaults" | grep -Eq \
    'prepared DS4 GGUF not found|requires.*METAL'
! printf '%s\n' "$http_defaults" | grep -q 'serve requires --stdio'

set +e
transport_conflict=$(env -u SNAP -u CHAT -u PROMPT -u SERVE \
    ./floyd serve --model "$model" --stdio --host 127.0.0.1 2>&1)
transport_status=$?
set -e
[ "$transport_status" -eq 2 ]
printf '%s\n' "$transport_conflict" | grep -q \
    'HTTP flags cannot be used with --stdio'

for port in 0 65536; do
    set +e
    bad_port=$(env -u SNAP -u CHAT -u PROMPT -u SERVE \
        ./floyd serve --model "$model" --port "$port" 2>&1)
    port_status=$?
    set -e
    [ "$port_status" -eq 2 ]
    printf '%s\n' "$bad_port" | grep -q \
        -- '--port must be an integer in 1..65535'
done

for command in chat run; do
    prompt_args=
    [ "$command" = chat ] || prompt_args='--prompt hello'
    for flag in '--host 127.0.0.1' '--port 8081' '--api-key secret' \
                '--served-model-name test-model'; do
        set +e
        # The intentional word splitting expands each flag and its value.
        # shellcheck disable=SC2086
        http_only=$(env -u SNAP -u CHAT -u PROMPT -u SERVE \
            ./floyd "$command" --model "$model" $prompt_args $flag 2>&1)
        http_only_status=$?
        set -e
        [ "$http_only_status" -eq 2 ]
        printf '%s\n' "$http_only" | grep -q \
            'HTTP flags are only valid for serve'
    done
done

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
