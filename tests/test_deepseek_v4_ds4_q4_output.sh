#!/bin/sh
set -eu

checkpoint=$1
q8_model=$2
q4_model=$3
tokens=${NGEN:-4}
tmp=$(mktemp -d -t floyd-deepseek-v4-ds4-q4-output.XXXXXX)
trap 'rm -rf "$tmp"' EXIT INT TERM

run_model() {
    name=$1
    model=$2
    set +e
    printf '%s\n:exit\n' '请写一个简短的中文科幻故事。' |
        env FLOYD_DEEPSEEK_V4_DS4_GGUF="$model" DSPARK_SPEC=0 \
            DEEPSEEK_V4_CHAT_TRACE=1 \
            ./floyd --model "$checkpoint" --ctx 128 --ngen "$tokens" \
            >"$tmp/$name.stdout" 2>"$tmp/$name.stderr"
    status=$?
    set -e
    if test "$status" -ne 0; then
        cat "$tmp/$name.stderr" "$tmp/$name.stdout"
        return "$status"
    fi
    grep -q 'DEEPSEEK_V4_BACKEND backend=metal-ds4' "$tmp/$name.stderr"
    grep -Fq "gguf=$model" "$tmp/$name.stderr"
    sed -n 's/^DEEPSEEK_V4_TOKEN //p' "$tmp/$name.stderr" >"$tmp/$name.tokens"
    test "$(wc -l <"$tmp/$name.tokens" | tr -d ' ')" -eq "$tokens"
}

run_model q8 "$q8_model"
run_model q4 "$q4_model"
diff -u "$tmp/q8.tokens" "$tmp/q4.tokens"

q8_perf=$(grep 'DEEPSEEK_V4_PERF ' "$tmp/q8.stderr" | tail -1)
q4_perf=$(grep 'DEEPSEEK_V4_PERF ' "$tmp/q4.stderr" | tail -1)
printf '%s\n%s\n' "$q8_perf" "$q4_perf"
printf 'DeepSeek V4 DS4 Q4 output parity: %s/%s tokens\n' "$tokens" "$tokens"
