#!/bin/sh
set -eu

model=$1
oracle=$2
python=${PYTHON:-python3}
expected=$($python - "$oracle" <<'PY'
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt") as source:
    print(int(source.get_tensor("final.argmax")[0]))
PY
)
output=$(PROMPT=hello NGEN=1 V4_CHAT_TRACE=1 \
    ./v4_chat "$model" 64 1 2>&1)
printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q "V4_TOKEN $expected"
backend=${EXPECT_BACKEND:-cpu}
printf '%s\n' "$output" | grep -q "V4_BACKEND backend=$backend"
printf '%s\n' "$output" | grep -q "V4_MATMUL metal_calls="
if [ "$backend" = metal ]; then
    printf '%s\n' "$output" | grep -Eq "V4_MATMUL metal_calls=[1-9][0-9]*"
fi
