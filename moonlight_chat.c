#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "json.h"
#include "tok.h"
#include "tok_unicode.h"
#include "tok_moon.h"
#include "moonlight_chat.h"
#include "moonlight_metal.h"

typedef struct {
    float probability;
    int token;
} MoonlightCandidate;

typedef struct {
    MoonlightModel *model;
    MoonlightSession *session;
    MoonlightModelInfo info;
    MoonlightChatOptions options;
    MTok tokenizer;
    float *logits;
    MoonlightCandidate *candidates;
    int *history;
    int history_length;
    uint64_t rng;
} MoonlightChat;

static double monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static char *read_file(const char *path, long *size) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) || (*size = ftell(file)) <= 0 ||
        *size > (1 << 20) || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return NULL;
    }
    char *text = malloc((size_t)*size + 1);
    if (!text || fread(text, 1, (size_t)*size, file) != (size_t)*size) {
        free(text);
        fclose(file);
        return NULL;
    }
    fclose(file);
    text[*size] = 0;
    return text;
}

int moonlight_metal_model_dir(const char *model_dir) {
    char path[4096];
    long size;
    if (!model_dir ||
        snprintf(path, sizeof(path), "%s/config.json", model_dir) >=
            (int)sizeof(path)) return 0;
    char *text = read_file(path, &size);
    if (!text) return 0;
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    jval *model_type = json_get(root, "model_type");
    int matches = model_type && model_type->t == J_STR &&
                  !strcmp(model_type->str, "deepseek_v3");
    free(arena);
    free(text);
    if (!matches ||
        snprintf(path, sizeof(path), "%s/tiktoken.model", model_dir) >=
            (int)sizeof(path)) return 0;
    FILE *tokenizer = fopen(path, "rb");
    if (!tokenizer) return 0;
    fclose(tokenizer);
    return 1;
}

static int candidate_descending(const void *left, const void *right) {
    float a = ((const MoonlightCandidate *)left)->probability;
    float b = ((const MoonlightCandidate *)right)->probability;
    return a < b ? 1 : a > b ? -1 : 0;
}

static double random_unit(MoonlightChat *chat) {
    chat->rng ^= chat->rng << 13;
    chat->rng ^= chat->rng >> 7;
    chat->rng ^= chat->rng << 17;
    return (double)(chat->rng >> 11) * (1.0 / 9007199254740992.0);
}

static int choose_token(MoonlightChat *chat) {
    int vocabulary = chat->info.vocab_size;
    if (chat->options.temperature <= 0.0f) {
        int best = 0;
        for (int token = 1; token < vocabulary; ++token)
            if (chat->logits[token] > chat->logits[best]) best = token;
        return best;
    }
    float maximum = chat->logits[0];
    for (int token = 1; token < vocabulary; ++token)
        if (chat->logits[token] > maximum) maximum = chat->logits[token];
    double total = 0.0;
    for (int token = 0; token < vocabulary; ++token) {
        float probability = expf((chat->logits[token] - maximum) /
                                 chat->options.temperature);
        chat->candidates[token].probability = probability;
        chat->candidates[token].token = token;
        total += probability;
    }
    for (int token = 0; token < vocabulary; ++token)
        chat->candidates[token].probability /= (float)total;
    qsort(chat->candidates, (size_t)vocabulary,
          sizeof(*chat->candidates), candidate_descending);
    double kept = 0.0;
    int count = 0;
    do {
        kept += chat->candidates[count++].probability;
    } while (count < vocabulary && kept < chat->options.top_p);
    double sample = random_unit(chat) * kept;
    double cumulative = 0.0;
    for (int index = 0; index < count; ++index) {
        cumulative += chat->candidates[index].probability;
        if (cumulative >= sample) return chat->candidates[index].token;
    }
    return chat->candidates[count - 1].token;
}

static int forward_pending(MoonlightChat *chat, char *error,
                           size_t error_size) {
    int position = moonlight_session_position(chat->session);
    int remaining = chat->history_length - position;
    int maximum_batch = chat->options.max_context < 128
        ? chat->options.max_context : 128;
    while (remaining > 0) {
        int count = remaining < maximum_batch ? remaining : maximum_batch;
        int ok = position == 0
            ? moonlight_session_prefill(chat->session,
                                        chat->history + position, count,
                                        chat->logits, error, error_size)
            : moonlight_session_append(chat->session,
                                       chat->history + position, count,
                                       chat->logits, error, error_size);
        if (!ok) return 0;
        position += count;
        remaining -= count;
    }
    return 1;
}

static void print_perf(MoonlightChat *chat, MoonlightStats before,
                       int generated) {
    MoonlightStats after = moonlight_session_stats(chat->session);
    uint64_t prompt_tokens = after.prefill_tokens - before.prefill_tokens;
    double prompt_ms = after.prefill_ms - before.prefill_ms;
    double decode_ms = after.decode_ms - before.decode_ms;
    fprintf(stderr,
            "MOONLIGHT_PERF prompt_tokens=%llu prompt_tps=%.3f "
            "generated_tokens=%d decode_tps=%.3f commands=%llu "
            "cpu_fallbacks=%llu\n",
            (unsigned long long)prompt_tokens,
            prompt_ms > 0.0 ? 1000.0 * prompt_tokens / prompt_ms : 0.0,
            generated,
            decode_ms > 0.0 ? 1000.0 * generated / decode_ms : 0.0,
            (unsigned long long)(after.command_buffers - before.command_buffers),
            (unsigned long long)after.cpu_fallbacks);
}

static int generate(MoonlightChat *chat, int eos, char *error,
                    size_t error_size) {
    MoonlightStats before = moonlight_session_stats(chat->session);
    if (!forward_pending(chat, error, error_size)) return -1;
    int generated = 0;
    for (; generated < chat->options.max_new_tokens; ++generated) {
        int token = choose_token(chat);
        if (chat->history_length >= chat->options.max_context) {
            snprintf(error, error_size, "Moonlight context is full");
            return -1;
        }
        chat->history[chat->history_length++] = token;
        if (token == eos) break;
        char piece[512];
        int size = mtok_decode(&chat->tokenizer, &token, 1,
                               piece, (int)sizeof(piece) - 1);
        if (size > 0) fwrite(piece, 1, (size_t)size, stdout);
        fflush(stdout);
        if (!moonlight_session_decode(chat->session, token, chat->logits,
                                      error, error_size)) return -1;
    }
    print_perf(chat, before, generated);
    return generated;
}

static int append_turn(MoonlightChat *chat, const char *prompt) {
    int length = chat->history_length;
    int limit = chat->options.max_context - chat->options.max_new_tokens - 1;
    if (length == 0)
        length = mtok_tmpl_msg(&chat->tokenizer, "system",
                               chat->options.system_prompt,
                               chat->history, length, limit);
    length = mtok_tmpl_msg(&chat->tokenizer, "user", prompt,
                           chat->history, length, limit);
    length = mtok_tmpl_genprompt(&chat->tokenizer, chat->history,
                                 length, limit);
    if (length < 0) return 0;
    chat->history_length = length;
    return 1;
}

static int open_chat(MoonlightChat *chat, const MoonlightChatOptions *options,
                     char *error, size_t error_size) {
    memset(chat, 0, sizeof(*chat));
    chat->options = *options;
    double start = monotonic_seconds();
    if (!moonlight_model_open(&chat->model, options->model_dir,
                              error, error_size)) return 0;
    chat->info = moonlight_model_info(chat->model);
    MoonlightOptions session_options = {
        .context_size = options->max_context,
        .max_batch = options->max_context < 128 ? options->max_context : 128,
    };
    if (!moonlight_session_create(&chat->session, chat->model,
                                  &session_options, error, error_size)) return 0;
    mtok_load(&chat->tokenizer, options->model_dir);
    chat->logits = malloc((size_t)chat->info.vocab_size * sizeof(*chat->logits));
    chat->candidates = malloc((size_t)chat->info.vocab_size *
                              sizeof(*chat->candidates));
    chat->history = malloc((size_t)options->max_context * sizeof(*chat->history));
    if (!chat->logits || !chat->candidates || !chat->history) {
        snprintf(error, error_size, "out of memory creating Moonlight chat");
        return 0;
    }
    const char *seed = getenv("SEED");
    chat->rng = seed ? (uint64_t)strtoull(seed, NULL, 10)
                     : (uint64_t)(monotonic_seconds() * 1000000000.0);
    if (!chat->rng) chat->rng = UINT64_C(0x9e3779b97f4a7c15);
    fprintf(stderr,
            "MOONLIGHT_BACKEND backend=metal-moonlight device=%s "
            "load_ms=%.3f context=%d batch=%d\n",
            moonlight_device_name(chat->model),
            (monotonic_seconds() - start) * 1000.0,
            options->max_context, session_options.max_batch);
    return 1;
}

static void close_chat(MoonlightChat *chat) {
    free(chat->logits);
    free(chat->candidates);
    free(chat->history);
    moonlight_session_destroy(chat->session);
    moonlight_model_close(chat->model);
}

int moonlight_chat_run(const MoonlightChatOptions *options) {
    if (!options || !options->model_dir || !options->system_prompt ||
        options->max_context <= 0 || options->max_new_tokens <= 0 ||
        options->max_new_tokens >= options->max_context ||
        options->temperature < 0.0f || options->top_p <= 0.0f ||
        options->top_p > 1.0f) {
        fprintf(stderr, "invalid Moonlight chat options\n");
        return 2;
    }
    MoonlightChat chat;
    char error[1024] = {0};
    if (!open_chat(&chat, options, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        close_chat(&chat);
        return 1;
    }
    int eos = mtok_special(&chat.tokenizer, "<|im_end|>");
    if (eos < 0) {
        fprintf(stderr, "Moonlight tokenizer has no <|im_end|> token\n");
        close_chat(&chat);
        return 1;
    }
    const char *one_shot = getenv("PROMPT");
    if (one_shot) {
        int generated = -1;
        if (!append_turn(&chat, one_shot))
            snprintf(error, sizeof(error), "Moonlight prompt is too long");
        else
            generated = generate(&chat, eos, error, sizeof(error));
        if (generated < 0) fprintf(stderr, "%s\n", error);
        putchar('\n');
        int status = generated < 0;
        close_chat(&chat);
        return status;
    }
    fprintf(stderr,
            "floyd chat [Moonlight] - :reset clears context, :exit quits\n");
    char *line = NULL;
    size_t capacity = 0;
    int status = 0;
    for (;;) {
        fputs("\n\xe2\x80\xba ", stdout);
        fflush(stdout);
        ssize_t size = getline(&line, &capacity, stdin);
        if (size < 0) break;
        if (size && line[size - 1] == '\n') line[--size] = 0;
        if (!strcmp(line, ":exit") || !strcmp(line, "/exit")) break;
        if (!strcmp(line, ":reset") || !strcmp(line, "/clear")) {
            moonlight_session_reset(chat.session);
            chat.history_length = 0;
            puts("context cleared");
            continue;
        }
        if (!size) continue;
        if (!append_turn(&chat, line)) {
            fprintf(stderr, "Moonlight context is full; use :reset\n");
            continue;
        }
        fputs("\xe2\x97\x86 ", stdout);
        fflush(stdout);
        if (generate(&chat, eos, error, sizeof(error)) < 0) {
            fprintf(stderr, "%s\n", error);
            status = 1;
            break;
        }
        putchar('\n');
    }
    free(line);
    close_chat(&chat);
    return status;
}
