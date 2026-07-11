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

static int v4_chat_turn(V4Runtime *runtime, Tok *tokenizer, const char *text,
                        int first_turn, int max_new_tokens, int close_turn) {
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
    if (close_turn) {
        int available = runtime->max_context
                      - runtime->state.layers[0].next_position - 1;
        if (max_new_tokens > available) max_new_tokens = available;
    }
    if (max_new_tokens <= 0) return -1;
    return v4_chat_generate(runtime, tokenizer, logits,
                            max_new_tokens, close_turn);
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
    if (max_context <= 0 || max_new_tokens <= 0) {
        fprintf(stderr, "max_context and max_new_tokens must be positive\n");
        return 2;
    }

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
                                  max_new_tokens, 0);
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
                                  max_new_tokens, 1);
        putchar('\n');
        if (result < 0) { status = 1; break; }
        first_turn = 0;
    }
    free(line);
    v4_runtime_free(&runtime);
    return status;
}
