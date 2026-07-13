#!/bin/sh
set -eu

model=${1:?usage: test_moonlight_oracle_runtime.sh MODEL REF}
reference=${2:?usage: test_moonlight_oracle_runtime.sh MODEL REF}
output=$(mktemp -d "${TMPDIR:-/tmp}/floyd-moonlight-oracle.XXXXXX")
trap 'rm -rf "$output"' EXIT

FLOYD_MOONLIGHT_ORACLE_DIR=$output ./floyd tf \
    --model "$model" --ref "$reference" --ebits 16 --dbits 16 >/dev/null 2>&1

for name in embed layer.0.input layer.0.input_norm layer.0.attn layer.0.post_attn \
    layer.0.post_norm layer.0.mlp layer.0.output layer.1.router_scores \
    layer.1.routed_mlp layer.1.shared_mlp layer.1.mlp final_norm logits
do
    test -s "$output/$name.f32"
done

printf 'Moonlight CPU runtime oracle: ok\n'
