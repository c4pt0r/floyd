#!/bin/sh
set -eu

help=$(./floyd help)
printf '%s\n' "$help" | grep -q '^usage:'
printf '%s\n' "$help" | grep -q '^  chat '
printf '%s\n' "$help" | grep -q '^  run '
printf '%s\n' "$help" | grep -q '^  serve '
printf '%s\n' "$help" | grep -q '^  help '

for removed in tf gen; do
    if printf '%s\n' "$help" | grep -q "^  $removed "; then
        echo "$removed is still advertised" >&2
        exit 1
    fi
    if ./floyd "$removed" >/dev/null 2>&1; then
        echo "$removed is still accepted" >&2
        exit 1
    fi
done

if printf '%s\n' "$help" | grep -Eq 'legacy|--ref|--cap|--ebits|--dbits|--metal'; then
    echo "legacy CPU/oracle flags are still advertised" >&2
    exit 1
fi

link_plan=$(make -Bn floyd)
if printf '%s\n' "$link_plan" | grep -Eq \
    'moonlight_oracle\.o|backend_metal\.o|deepseek_v4_ggml\.o|libllama'; then
    echo "floyd still links a temporary oracle or non-resident fallback backend" >&2
    exit 1
fi

if nm floyd | grep -Eq \
    '_model_init$|_forward_all$|_fm_init$|_moonlight_oracle_|_deepseek_v4_ggml_'; then
    echo "floyd still contains a legacy CPU/oracle/fallback runtime" >&2
    exit 1
fi

if nm floyd | grep -Eq \
    '_cpu_decode_scratch_init$|_forward_first_token_cpu$|_prefill_layer_major_cpu$|_session_cpu_reset_cache$'; then
    echo "floyd still contains the DS4 CPU inference backend" >&2
    exit 1
fi

for path in \
    moonlight_oracle.c \
    moonlight_oracle.h \
    tests/test_moonlight_oracle.c \
    tests/test_moonlight_oracle.py \
    tests/test_moonlight_oracle_runtime.sh \
    tools/make_moonlight_oracle.py; do
    if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
        echo "temporary oracle is still tracked: $path" >&2
        exit 1
    fi
done

if grep -Eq 'COLI_CUDA|model_init\(|forward_all\(' floyd.c; then
    echo "legacy CPU/CUDA runtime remains in the product entry point" >&2
    exit 1
fi

printf 'Metal-only build contract: ok\n'
