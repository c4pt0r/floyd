#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tok.h"
#include "v4_chat_format.h"
#include "v4_runtime.h"

static int v4_chat_trace_enabled(void) {
    const char *value = getenv("V4_CHAT_TRACE");
    return value && atoi(value) != 0;
}

static int v4_chat_backend_init(void) {
    int requested = getenv("FLOYD_METAL") && atoi(getenv("FLOYD_METAL")) != 0;
#ifdef FLOYD_METAL
    if (requested) {
        int min_batch = getenv("FM_MIN_S") ? atoi(getenv("FM_MIN_S")) : 8;
        if (!v4_quant_backend_enable_metal(min_batch)) {
            fprintf(stderr, "V4 Metal requested but unavailable or FM_MIN_S < 2\n");
            return 0;
        }
        fprintf(stderr, "V4_BACKEND backend=metal device=%s min_batch=%d\n",
                fm_device_name(), v4_quant_backend_min_batch());
    } else {
        fprintf(stderr, "V4_BACKEND backend=cpu\n");
    }
#else
    if (requested) {
        fprintf(stderr, "FLOYD_METAL requires: make METAL=1 v4_chat\n");
        return 0;
    }
    fprintf(stderr, "V4_BACKEND backend=cpu\n");
#endif
    return 1;
}

static void v4_chat_print_backend_stats(V4QuantBackendStats before) {
    V4QuantBackendStats after = v4_quant_backend_stats();
    fprintf(stderr, "V4_MATMUL metal_calls=%llu cpu_fallbacks=%llu\n",
            (unsigned long long)(after.metal_calls - before.metal_calls),
            (unsigned long long)(after.cpu_fallbacks - before.cpu_fallbacks));
}

static int v4_chat_emit(Tok *tokenizer, int token_id) {
    char bytes[4096];
    int count = tok_decode(tokenizer, &token_id, 1, bytes, sizeof(bytes) - 1);
    if (count < 0 || count >= (int)sizeof(bytes)) return 0;
    if (count && fwrite(bytes, 1, (size_t)count, stdout) != (size_t)count)
        return 0;
    fflush(stdout);
    return 1;
}

static int v4_chat_generate(V4Runtime *runtime, Tok *tokenizer,
                            const float *logits, int max_new_tokens,
                            int close_turn) {
    int eos = tok_id_of(tokenizer, "<｜end▁of▁sentence｜>");
    if (eos < 0 || !logits || max_new_tokens <= 0) return 0;
    int emitted = 0, ended = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        int token = v4_runtime_argmax(logits);
        if (token < 0) return -1;
        if (v4_chat_trace_enabled())
            fprintf(stderr, "V4_TOKEN %d\n", token);
        if (token == eos) {
            ended = 1;
            break;
        }
        if (!v4_chat_emit(tokenizer, token)) return -1;
        emitted++;
        if (step + 1 < max_new_tokens || close_turn) {
            if (!v4_runtime_forward(runtime, &token, 1, &logits)) return -1;
        }
    }
    if (close_turn) {
        if (!ended && runtime->state.layers[0].next_position
                          >= runtime->max_context)
            return -1;
        if (!v4_runtime_forward(runtime, &eos, 1, &logits)) return -1;
    }
    return emitted;
}

static int v4_chat_generate_spec(V4Runtime *runtime, Tok *tokenizer,
                                 const float *logits, int max_new_tokens,
                                 int close_turn) {
    int eos = tok_id_of(tokenizer, "<｜end▁of▁sentence｜>");
    if (eos < 0 || !logits || max_new_tokens <= 0) return 0;
    int emitted = 0, accepted = 0, proposed = 0;
    int pending = -1, ended = 0;

    int token = v4_runtime_argmax(logits);
    if (token < 0) return -1;
    if (v4_chat_trace_enabled()) fprintf(stderr, "V4_TOKEN %d\n", token);
    if (token == eos) {
        ended = 1;
    } else {
        if (!v4_chat_emit(tokenizer, token)) return -1;
        emitted++;
        pending = token;
    }

    while (!ended && emitted < max_new_tokens) {
        if (!v4_runtime_forward(runtime, &pending, 1, &logits)) return -1;
        pending = -1;
        int input_id = v4_runtime_argmax(logits);
        int64_t proposals[6];
        float confidence[5];
        if (!v4_runtime_dspark_propose(
                runtime, input_id, proposals, confidence)) return -1;
        for (int rank = 0; rank < 6 && emitted < max_new_tokens; rank++) {
            int match = 0;
            int base = v4_runtime_verify_token(logits, proposals[rank], &match);
            if (base < 0) return -1;
            proposed++;
            if (match) accepted++;
            if (v4_chat_trace_enabled())
                fprintf(stderr, "V4_TOKEN %d\n", base);
            if (base == eos) {
                ended = 1;
                break;
            }
            if (!v4_chat_emit(tokenizer, base)) return -1;
            emitted++;
            pending = base;
            if (!match || emitted == max_new_tokens) break;
            if (rank + 1 < 6) {
                if (!v4_runtime_forward(runtime, &pending, 1, &logits)) return -1;
                pending = -1;
            }
        }
    }
    fprintf(stderr, "V4_SPEC %d/%d\n", accepted, proposed);
    if (close_turn) {
        if (pending >= 0 &&
            !v4_runtime_forward(runtime, &pending, 1, &logits)) return -1;
        if (!v4_runtime_forward(runtime, &eos, 1, &logits)) return -1;
    }
    return emitted;
}

static int v4_chat_turn(V4Runtime *runtime, Tok *tokenizer, const char *text,
                        int first_turn, int max_new_tokens, int close_turn,
                        int use_spec) {
    V4QuantBackendStats stats_before = v4_quant_backend_stats();
    size_t text_size = strlen(text);
    size_t prompt_cap = text_size + 256;
    char *prompt = malloc(prompt_cap);
    if (!prompt) return -1;
    size_t prompt_size = v4_chat_append_user(
        prompt, prompt_cap, 0, text, first_turn);
    if (prompt_size == SIZE_MAX) { free(prompt); return -1; }

    int remaining = runtime->max_context
                  - runtime->state.layers[0].next_position;
    int *ids = malloc((size_t)(remaining + 1) * sizeof(int));
    if (!ids) { free(prompt); return -1; }
    int count = tok_encode(tokenizer, prompt, (int)prompt_size,
                           ids, remaining + 1);
    free(prompt);
    if (count <= 0 || count > remaining) {
        fprintf(stderr, "V4 chat context exhausted by prompt\n");
        free(ids);
        return -1;
    }
    const float *logits = NULL;
    int ok = v4_runtime_forward(runtime, ids, count, &logits);
    free(ids);
    if (!ok) return -1;
    if (use_spec && !runtime->dspark_ready &&
        !v4_runtime_dspark_prefill(runtime)) return -1;
    if (close_turn) {
        int available = runtime->max_context
                      - runtime->state.layers[0].next_position - 1;
        if (max_new_tokens > available) max_new_tokens = available;
    }
    if (max_new_tokens <= 0) return -1;
    int result = use_spec
         ? v4_chat_generate_spec(runtime, tokenizer, logits,
                                 max_new_tokens, close_turn)
         : v4_chat_generate(runtime, tokenizer, logits,
                            max_new_tokens, close_turn);
    v4_chat_print_backend_stats(stats_before);
    return result;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr,
                "usage: %s <DeepSeek-V4-Flash-DSpark> [max_context] [max_new_tokens]\n",
                argv[0]);
        return 2;
    }
    int max_context = argc >= 3 ? atoi(argv[2]) : 512;
    int max_new_tokens = argc >= 4 ? atoi(argv[3]) : 16;
    if (getenv("NGEN")) max_new_tokens = atoi(getenv("NGEN"));
    int use_spec = getenv("DSPARK_SPEC") && atoi(getenv("DSPARK_SPEC")) != 0;
    if (max_context <= 0 || max_new_tokens <= 0) {
        fprintf(stderr, "max_context and max_new_tokens must be positive\n");
        return 2;
    }
    if (!v4_chat_backend_init()) return 2;

    char tokenizer_path[2048];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", argv[1]);
    Tok tokenizer;
    tok_load(&tokenizer, tokenizer_path);
    V4Runtime runtime;
    if (!v4_runtime_init(&runtime, argv[1], max_context)) {
        fprintf(stderr, "failed to initialize V4 runtime\n");
        return 1;
    }

    const char *one_shot = getenv("PROMPT");
    if (one_shot) {
        int result = v4_chat_turn(&runtime, &tokenizer, one_shot, 1,
                                  max_new_tokens, 0, use_spec);
        if (result >= 0) putchar('\n');
        v4_runtime_free(&runtime);
        return result < 0;
    }

    puts("DeepSeek V4 chat (/clear, /exit)");
    char *line = NULL;
    size_t capacity = 0;
    int first_turn = 1, status = 0;
    for (;;) {
        fputs("user> ", stdout);
        fflush(stdout);
        ssize_t size = getline(&line, &capacity, stdin);
        if (size < 0) break;
        if (size && line[size - 1] == '\n') line[--size] = 0;
        if (!strcmp(line, "/exit")) break;
        if (!strcmp(line, "/clear")) {
            v4_runtime_reset(&runtime);
            first_turn = 1;
            puts("context cleared");
            continue;
        }
        if (!size) continue;
        fputs("assistant> ", stdout);
        fflush(stdout);
        int result = v4_chat_turn(&runtime, &tokenizer, line, first_turn,
                                  max_new_tokens, 1, use_spec);
        putchar('\n');
        if (result < 0) { status = 1; break; }
        first_turn = 0;
    }
    free(line);
    v4_runtime_free(&runtime);
    return status;
}
