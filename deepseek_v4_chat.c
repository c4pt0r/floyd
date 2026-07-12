#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_chat.h"
#include "tok.h"
#include "deepseek_v4_chat_format.h"
#include "deepseek_v4_runtime.h"
#include "json.h"

#ifdef FLOYD_DEEPSEEK_V4_GGML
#include "deepseek_v4_ggml.h"
#endif
#ifdef FLOYD_DEEPSEEK_V4_DS4
#include "deepseek_v4_ds4.h"
#endif

int deepseek_v4_model_dir(const char *model_dir) {
    if (!model_dir) return 0;
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) || ftell(file) <= 0) {
        fclose(file);
        return 0;
    }
    long size = ftell(file);
    if (size > (1 << 20) || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return 0;
    }
    char *text = malloc((size_t)size + 1);
    if (!text) { fclose(file); return 0; }
    size_t read_size = fread(text, 1, (size_t)size, file);
    fclose(file);
    if (read_size != (size_t)size) { free(text); return 0; }
    text[size] = 0;
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    jval *model_type = json_get(root, "model_type");
    int is_v4 = model_type && model_type->t == J_STR &&
                !strcmp(model_type->str, "deepseek_v4");
    free(arena);
    free(text);
    return is_v4;
}

static int deepseek_v4_chat_trace_enabled(void) {
    const char *value = getenv("DEEPSEEK_V4_CHAT_TRACE");
    return value && atoi(value) != 0;
}

#ifdef FLOYD_DEEPSEEK_V4_DS4
static int deepseek_v4_chat_ds4_emit(int token, const char *piece,
                                    size_t piece_size, void *user_data) {
    (void)user_data;
    if (deepseek_v4_chat_trace_enabled())
        fprintf(stderr, "DEEPSEEK_V4_TOKEN %d\n", token);
    if (piece_size && fwrite(piece, 1, piece_size, stdout) != piece_size) return 0;
    fflush(stdout);
    return 1;
}

static void deepseek_v4_chat_ds4_perf(const DeepSeekV4Ds4Stats *stats) {
    double prompt_tps = stats->prompt_ms > 0.0
        ? 1000.0 * stats->prompt_tokens / stats->prompt_ms : 0.0;
    double decode_tps = stats->decode_ms > 0.0
        ? 1000.0 * stats->generated_tokens / stats->decode_ms : 0.0;
    fprintf(stderr,
            "DEEPSEEK_V4_PERF load_ms=%.3f prompt_tokens=%d prompt_tps=%.3f "
            "generated_tokens=%d decode_tps=%.3f spec_rounds=%d spec_tokens=%d\n",
            stats->load_ms, stats->prompt_tokens, prompt_tps,
            stats->generated_tokens, decode_tps,
            stats->speculative_rounds, stats->speculative_tokens);
}

static int deepseek_v4_chat_run_ds4(const DeepSeekV4ChatOptions *options,
                                    const char *model_path) {
    char error[4096];
    fprintf(stderr, "DEEPSEEK_V4_BACKEND backend=%s gguf=%s\n",
            deepseek_v4_ds4_backend_name(), model_path);
    if (options->use_spec && !getenv("FLOYD_DEEPSEEK_V4_DS4_MTP"))
        fprintf(stderr, "DEEPSEEK_V4_SPEC disabled=mtp-not-prepared\n");

    DeepSeekV4Ds4Session *session = deepseek_v4_ds4_open(
        model_path, options->max_context, options->use_spec,
        error, sizeof(error));
    if (!session) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    const char *one_shot = getenv("PROMPT");
    if (one_shot) {
        DeepSeekV4Ds4Stats stats;
        int generated = deepseek_v4_ds4_generate_user(
            session, one_shot, options->max_new_tokens,
            deepseek_v4_chat_ds4_emit, NULL, &stats, error, sizeof(error));
        if (generated < 0) fprintf(stderr, "%s\n", error);
        else deepseek_v4_chat_ds4_perf(&stats);
        putchar('\n');
        deepseek_v4_ds4_close(session);
        return generated < 0;
    }

    fprintf(stderr, "floyd chat [DeepSeek V4] — :reset azzera, :exit esce\n");
    char *line = NULL;
    size_t capacity = 0;
    int status = 0;
    for (;;) {
        fputs("\n› ", stdout);
        fflush(stdout);
        ssize_t size = getline(&line, &capacity, stdin);
        if (size < 0) break;
        if (size && line[size - 1] == '\n') line[--size] = 0;
        if (!strcmp(line, ":exit") || !strcmp(line, "/exit")) break;
        if (!strcmp(line, ":reset") || !strcmp(line, "/clear")) {
            if (!deepseek_v4_ds4_reset(session)) {
                fprintf(stderr, "failed to reset DeepSeek V4 DS4 state\n");
                status = 1;
                break;
            }
            puts("context cleared");
            continue;
        }
        if (!size) continue;
        fputs("◆ ", stdout);
        fflush(stdout);
        DeepSeekV4Ds4Stats stats;
        int generated = deepseek_v4_ds4_generate_user(
            session, line, options->max_new_tokens,
            deepseek_v4_chat_ds4_emit, NULL, &stats, error, sizeof(error));
        putchar('\n');
        if (generated < 0) {
            fprintf(stderr, "%s\n", error);
            status = 1;
            break;
        }
        deepseek_v4_chat_ds4_perf(&stats);
    }
    free(line);
    deepseek_v4_ds4_close(session);
    return status;
}
#endif

#ifdef FLOYD_DEEPSEEK_V4_GGML
static int deepseek_v4_chat_ggml_emit(int token, const char *piece,
                                     size_t piece_size, void *user_data) {
    (void)user_data;
    if (deepseek_v4_chat_trace_enabled())
        fprintf(stderr, "DEEPSEEK_V4_TOKEN %d\n", token);
    if (piece_size && fwrite(piece, 1, piece_size, stdout) != piece_size) return 0;
    fflush(stdout);
    return 1;
}

static void deepseek_v4_chat_ggml_perf(const DeepSeekV4GgmlStats *stats) {
    double prompt_tps = stats->prompt_ms > 0.0
        ? 1000.0 * stats->prompt_tokens / stats->prompt_ms : 0.0;
    double decode_tps = stats->decode_ms > 0.0
        ? 1000.0 * stats->generated_tokens / stats->decode_ms : 0.0;
    fprintf(stderr,
            "DEEPSEEK_V4_PERF load_ms=%.3f prompt_tokens=%d prompt_tps=%.3f "
            "generated_tokens=%d decode_tps=%.3f\n",
            stats->load_ms, stats->prompt_tokens, prompt_tps,
            stats->generated_tokens, decode_tps);
}

static int deepseek_v4_chat_run_ggml(const DeepSeekV4ChatOptions *options) {
    char model_path[4096], error[4096];
    if (!deepseek_v4_ggml_find_model(options->model_dir, model_path,
                                     sizeof(model_path), error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 2;
    }
    fprintf(stderr, "DEEPSEEK_V4_BACKEND backend=%s gguf=%s\n",
            deepseek_v4_ggml_backend_name(), model_path);
    if (options->use_spec)
        fprintf(stderr, "DEEPSEEK_V4_SPEC disabled=mtp-not-prepared\n");

    DeepSeekV4GgmlSession *session = deepseek_v4_ggml_open(
        options->model_dir, options->max_context, error, sizeof(error));
    if (!session) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    const char *one_shot = getenv("PROMPT");
    if (one_shot) {
        DeepSeekV4GgmlStats stats;
        int generated = deepseek_v4_ggml_generate_user(
            session, one_shot, 1, options->max_new_tokens,
            deepseek_v4_chat_ggml_emit, NULL, &stats, error, sizeof(error));
        if (generated < 0) fprintf(stderr, "%s\n", error);
        else deepseek_v4_chat_ggml_perf(&stats);
        putchar('\n');
        deepseek_v4_ggml_close(session);
        return generated < 0;
    }

    fprintf(stderr, "floyd chat [DeepSeek V4] — :reset azzera, :exit esce\n");
    char *line = NULL;
    size_t capacity = 0;
    int first_turn = 1, status = 0;
    for (;;) {
        fputs("\n› ", stdout);
        fflush(stdout);
        ssize_t size = getline(&line, &capacity, stdin);
        if (size < 0) break;
        if (size && line[size - 1] == '\n') line[--size] = 0;
        if (!strcmp(line, ":exit") || !strcmp(line, "/exit")) break;
        if (!strcmp(line, ":reset") || !strcmp(line, "/clear")) {
            if (!deepseek_v4_ggml_reset(session)) {
                fprintf(stderr, "failed to reset DeepSeek V4 ggml state\n");
                status = 1;
                break;
            }
            first_turn = 1;
            puts("context cleared");
            continue;
        }
        if (!size) continue;
        fputs("◆ ", stdout);
        fflush(stdout);
        DeepSeekV4GgmlStats stats;
        int generated = deepseek_v4_ggml_generate_user(
            session, line, first_turn, options->max_new_tokens,
            deepseek_v4_chat_ggml_emit, NULL, &stats, error, sizeof(error));
        putchar('\n');
        if (generated < 0) {
            fprintf(stderr, "%s\n", error);
            status = 1;
            break;
        }
        deepseek_v4_chat_ggml_perf(&stats);
        first_turn = 0;
    }
    free(line);
    deepseek_v4_ggml_close(session);
    return status;
}
#endif

static int deepseek_v4_chat_backend_init(void) {
    const char *metal_env = getenv("FLOYD_METAL");
#ifdef FLOYD_METAL
    int requested = !metal_env || atoi(metal_env) != 0;
    if (requested) {
        int min_batch = getenv("FM_MIN_S") ? atoi(getenv("FM_MIN_S")) : 8;
        if (!deepseek_v4_quant_backend_enable_metal(min_batch)) {
            fprintf(stderr, "DeepSeek V4 Metal requested but unavailable or FM_MIN_S < 2\n");
            return 0;
        }
        fprintf(stderr, "DEEPSEEK_V4_BACKEND backend=metal device=%s min_batch=%d\n",
                fm_device_name(), deepseek_v4_quant_backend_min_batch());
    } else {
        fprintf(stderr, "DEEPSEEK_V4_BACKEND backend=cpu\n");
    }
#else
    int requested = metal_env && atoi(metal_env) != 0;
    if (requested) {
        fprintf(stderr, "FLOYD_METAL requires: make METAL=1 floyd\n");
        return 0;
    }
    fprintf(stderr, "DEEPSEEK_V4_BACKEND backend=cpu\n");
#endif
    return 1;
}

static void deepseek_v4_chat_print_backend_stats(DeepSeekV4QuantBackendStats before) {
    DeepSeekV4QuantBackendStats after = deepseek_v4_quant_backend_stats();
    fprintf(stderr, "DEEPSEEK_V4_MATMUL metal_calls=%llu cpu_fallbacks=%llu\n",
            (unsigned long long)(after.metal_calls - before.metal_calls),
            (unsigned long long)(after.cpu_fallbacks - before.cpu_fallbacks));
}

static int deepseek_v4_chat_emit(Tok *tokenizer, int token_id) {
    char bytes[4096];
    int count = tok_decode(tokenizer, &token_id, 1, bytes, sizeof(bytes) - 1);
    if (count < 0 || count >= (int)sizeof(bytes)) return 0;
    if (count && fwrite(bytes, 1, (size_t)count, stdout) != (size_t)count)
        return 0;
    fflush(stdout);
    return 1;
}

static int deepseek_v4_chat_generate(DeepSeekV4Runtime *runtime, Tok *tokenizer,
                            const float *logits, int max_new_tokens,
                            int close_turn) {
    int eos = tok_id_of(tokenizer, "<｜end▁of▁sentence｜>");
    if (eos < 0 || !logits || max_new_tokens <= 0) return 0;
    int emitted = 0, ended = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        int token = deepseek_v4_runtime_argmax(logits);
        if (token < 0) return -1;
        if (deepseek_v4_chat_trace_enabled())
            fprintf(stderr, "DEEPSEEK_V4_TOKEN %d\n", token);
        if (token == eos) {
            ended = 1;
            break;
        }
        if (!deepseek_v4_chat_emit(tokenizer, token)) return -1;
        emitted++;
        if (step + 1 < max_new_tokens || close_turn) {
            if (!deepseek_v4_runtime_forward(runtime, &token, 1, &logits)) return -1;
        }
    }
    if (close_turn) {
        if (!ended && runtime->state.layers[0].next_position
                          >= runtime->max_context)
            return -1;
        if (!deepseek_v4_runtime_forward(runtime, &eos, 1, &logits)) return -1;
    }
    return emitted;
}

static int deepseek_v4_chat_generate_spec(DeepSeekV4Runtime *runtime, Tok *tokenizer,
                                 const float *logits, int max_new_tokens,
                                 int close_turn) {
    int eos = tok_id_of(tokenizer, "<｜end▁of▁sentence｜>");
    if (eos < 0 || !logits || max_new_tokens <= 0) return 0;
    int emitted = 0, accepted = 0, proposed = 0;
    int pending = -1, ended = 0;

    int token = deepseek_v4_runtime_argmax(logits);
    if (token < 0) return -1;
    if (deepseek_v4_chat_trace_enabled()) fprintf(stderr, "DEEPSEEK_V4_TOKEN %d\n", token);
    if (token == eos) {
        ended = 1;
    } else {
        if (!deepseek_v4_chat_emit(tokenizer, token)) return -1;
        emitted++;
        pending = token;
    }

    while (!ended && emitted < max_new_tokens) {
        if (!deepseek_v4_runtime_forward(runtime, &pending, 1, &logits)) return -1;
        pending = -1;
        int input_id = deepseek_v4_runtime_argmax(logits);
        int64_t proposals[6];
        float confidence[5];
        if (!deepseek_v4_runtime_dspark_propose(
                runtime, input_id, proposals, confidence)) return -1;
        for (int rank = 0; rank < 6 && emitted < max_new_tokens; rank++) {
            int match = 0;
            int base = deepseek_v4_runtime_verify_token(logits, proposals[rank], &match);
            if (base < 0) return -1;
            proposed++;
            if (match) accepted++;
            if (deepseek_v4_chat_trace_enabled())
                fprintf(stderr, "DEEPSEEK_V4_TOKEN %d\n", base);
            if (base == eos) {
                ended = 1;
                break;
            }
            if (!deepseek_v4_chat_emit(tokenizer, base)) return -1;
            emitted++;
            pending = base;
            if (!match || emitted == max_new_tokens) break;
            if (rank + 1 < 6) {
                if (!deepseek_v4_runtime_forward(runtime, &pending, 1, &logits)) return -1;
                pending = -1;
            }
        }
    }
    fprintf(stderr, "DEEPSEEK_V4_SPEC %d/%d\n", accepted, proposed);
    if (close_turn) {
        if (pending >= 0 &&
            !deepseek_v4_runtime_forward(runtime, &pending, 1, &logits)) return -1;
        if (!deepseek_v4_runtime_forward(runtime, &eos, 1, &logits)) return -1;
    }
    return emitted;
}

static int deepseek_v4_chat_turn(DeepSeekV4Runtime *runtime, Tok *tokenizer, const char *text,
                        int first_turn, int max_new_tokens, int close_turn,
                        int use_spec) {
    DeepSeekV4QuantBackendStats stats_before = deepseek_v4_quant_backend_stats();
    size_t text_size = strlen(text);
    size_t prompt_cap = text_size + 256;
    char *prompt = malloc(prompt_cap);
    if (!prompt) return -1;
    size_t prompt_size = deepseek_v4_chat_append_user(
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
        fprintf(stderr, "DeepSeek V4 chat context exhausted by prompt\n");
        free(ids);
        return -1;
    }
    const float *logits = NULL;
    int ok = deepseek_v4_runtime_forward(runtime, ids, count, &logits);
    free(ids);
    if (!ok) return -1;
    if (use_spec && !runtime->dspark_ready &&
        !deepseek_v4_runtime_dspark_prefill(runtime)) return -1;
    if (close_turn) {
        int available = runtime->max_context
                      - runtime->state.layers[0].next_position - 1;
        if (max_new_tokens > available) max_new_tokens = available;
    }
    if (max_new_tokens <= 0) return -1;
    int result = use_spec
         ? deepseek_v4_chat_generate_spec(runtime, tokenizer, logits,
                                 max_new_tokens, close_turn)
         : deepseek_v4_chat_generate(runtime, tokenizer, logits,
                            max_new_tokens, close_turn);
    deepseek_v4_chat_print_backend_stats(stats_before);
    return result;
}

int deepseek_v4_chat_run(const DeepSeekV4ChatOptions *options) {
    if (!options || !options->model_dir || options->max_context <= 0 ||
        options->max_new_tokens <= 0) {
        fprintf(stderr, "max_context and max_new_tokens must be positive\n");
        return 2;
    }
    const char *reference = getenv("FLOYD_DEEPSEEK_V4_REFERENCE");
    if (!reference || atoi(reference) == 0) {
#ifdef FLOYD_DEEPSEEK_V4_GGML
        /* An explicit exact-runtime model selection overrides DS4 auto-discovery. */
        if (getenv("FLOYD_DEEPSEEK_V4_GGUF"))
            return deepseek_v4_chat_run_ggml(options);
#endif
#ifdef FLOYD_DEEPSEEK_V4_DS4
        char ds4_model[4096], ds4_error[4096];
        if (deepseek_v4_ds4_find_model(options->model_dir, ds4_model,
                                       sizeof(ds4_model), ds4_error,
                                       sizeof(ds4_error)))
            return deepseek_v4_chat_run_ds4(options, ds4_model);
        if (getenv("FLOYD_DEEPSEEK_V4_DS4_GGUF")) {
            fprintf(stderr, "%s\n", ds4_error);
            return 2;
        }
#endif
#ifdef FLOYD_DEEPSEEK_V4_GGML
        return deepseek_v4_chat_run_ggml(options);
#else
        fprintf(stderr,
                "DeepSeek V4 chat requires the resident Metal runtime; "
                "rebuild with: make METAL=1 floyd\n");
        return 2;
#endif
    }
    if (!deepseek_v4_chat_backend_init()) return 2;

    char tokenizer_path[2048];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             options->model_dir);
    Tok tokenizer;
    tok_load(&tokenizer, tokenizer_path);
    DeepSeekV4Runtime runtime;
    if (!deepseek_v4_runtime_init(&runtime, options->model_dir,
                                  options->max_context)) {
        fprintf(stderr, "failed to initialize DeepSeek V4 runtime\n");
        return 1;
    }

    const char *one_shot = getenv("PROMPT");
    if (one_shot) {
        int result = deepseek_v4_chat_turn(&runtime, &tokenizer, one_shot, 1,
                                  options->max_new_tokens, 0,
                                  options->use_spec);
        if (result >= 0) putchar('\n');
        deepseek_v4_runtime_free(&runtime);
        return result < 0;
    }

    fprintf(stderr, "floyd chat [DeepSeek V4] — :reset azzera, :exit esce\n");
    char *line = NULL;
    size_t capacity = 0;
    int first_turn = 1, status = 0;
    for (;;) {
        fputs("\n› ", stdout);
        fflush(stdout);
        ssize_t size = getline(&line, &capacity, stdin);
        if (size < 0) break;
        if (size && line[size - 1] == '\n') line[--size] = 0;
        if (!strcmp(line, ":exit") || !strcmp(line, "/exit")) break;
        if (!strcmp(line, ":reset") || !strcmp(line, "/clear")) {
            deepseek_v4_runtime_reset(&runtime);
            first_turn = 1;
            puts("context cleared");
            continue;
        }
        if (!size) continue;
        fputs("◆ ", stdout);
        fflush(stdout);
        int result = deepseek_v4_chat_turn(&runtime, &tokenizer, line, first_turn,
                                  options->max_new_tokens, 1,
                                  options->use_spec);
        putchar('\n');
        if (result < 0) { status = 1; break; }
        first_turn = 0;
    }
    free(line);
    deepseek_v4_runtime_free(&runtime);
    return status;
}
