#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_chat.h"
#include "deepseek_v4_ds4.h"
#include "deepseek_v4_serve.h"
#include "json.h"
#include "openai_http.h"

int deepseek_v4_model_dir(const char *model_dir) {
    if (!model_dir) return 0;
    char path[2048];
    if (snprintf(path, sizeof(path), "%s/config.json", model_dir) >=
        (int)sizeof(path)) return 0;
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        if (file) fclose(file);
        return 0;
    }
    long size = ftell(file);
    if (size <= 0 || size > (1 << 20) || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return 0;
    }
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, file) != (size_t)size) {
        free(text);
        fclose(file);
        return 0;
    }
    fclose(file);
    text[size] = 0;
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    jval *model_type = json_get(root, "model_type");
    int matches = model_type && model_type->t == J_STR &&
                  !strcmp(model_type->str, "deepseek_v4");
    free(arena);
    free(text);
    return matches;
}

static int trace_enabled(void) {
    const char *value = getenv("DEEPSEEK_V4_CHAT_TRACE");
    return value && atoi(value) != 0;
}

static int emit_token(int token, const char *piece, size_t piece_size,
                      void *user_data) {
    (void)user_data;
    if (trace_enabled()) fprintf(stderr, "DEEPSEEK_V4_TOKEN %d\n", token);
    if (piece_size && fwrite(piece, 1, piece_size, stdout) != piece_size)
        return 0;
    fflush(stdout);
    return 1;
}

static void print_perf(const DeepSeekV4Ds4Stats *stats) {
    double prompt_tps = stats->prompt_ms > 0.0
        ? 1000.0 * stats->prompt_tokens / stats->prompt_ms : 0.0;
    double decode_tps = stats->decode_ms > 0.0
        ? 1000.0 * stats->generated_tokens / stats->decode_ms : 0.0;
    fprintf(stderr,
            "DEEPSEEK_V4_PERF load_ms=%.3f prompt_tokens=%d prompt_tps=%.3f "
            "generated_tokens=%d decode_tps=%.3f spec_rounds=%d "
            "spec_accepted=%d spec_proposed=%d direct_accepted=%d "
            "prefix1_accepted=%d frontier_snapshots=%d "
            "proposal_early_skips=%d target_ms=%.3f proposal_ms=%.3f "
            "verify_ms=%.3f verify_layer_encode_ms=%.3f "
            "verify_layer_execute_ms=%.3f verify_head_ms=%.3f "
            "verify_read_ms=%.3f replay_ms=%.3f\n",
            stats->load_ms, stats->prompt_tokens, prompt_tps,
            stats->generated_tokens, decode_tps,
            stats->speculative_rounds, stats->speculative_tokens,
            stats->speculative_proposed, stats->speculative_direct_accepted,
            stats->speculative_prefix1_accepted,
            stats->speculative_frontier_snapshots,
            stats->speculative_proposal_early_skips,
            stats->speculative_target_ms, stats->speculative_proposal_ms,
            stats->speculative_verify_ms,
            stats->speculative_verify_layer_encode_ms,
            stats->speculative_verify_layer_execute_ms,
            stats->speculative_verify_head_ms,
            stats->speculative_verify_read_ms,
            stats->speculative_replay_ms);
    fprintf(stderr,
            "DEEPSEEK_V4_SCHEDULE d1c1=%llu/%.3f d2c1=%llu/%.3f "
            "d2c2=%llu/%.3f d3c1=%llu/%.3f d3c2=%llu/%.3f "
            "d3c3=%llu/%.3f\n",
            stats->speculative_verify_outcome_calls[1][1],
            stats->speculative_verify_outcome_ms[1][1],
            stats->speculative_verify_outcome_calls[2][1],
            stats->speculative_verify_outcome_ms[2][1],
            stats->speculative_verify_outcome_calls[2][2],
            stats->speculative_verify_outcome_ms[2][2],
            stats->speculative_verify_outcome_calls[3][1],
            stats->speculative_verify_outcome_ms[3][1],
            stats->speculative_verify_outcome_calls[3][2],
            stats->speculative_verify_outcome_ms[3][2],
            stats->speculative_verify_outcome_calls[3][3],
            stats->speculative_verify_outcome_ms[3][3]);
}

int deepseek_v4_chat_run(const DeepSeekV4ChatOptions *options) {
    if (!options || !options->model_dir || options->max_context <= 0 ||
        options->max_new_tokens <= 0) {
        fprintf(stderr, "invalid DeepSeek V4 chat options\n");
        return 2;
    }
    char model_path[4096], error[4096] = {0};
    if (!deepseek_v4_ds4_find_model(options->model_dir, model_path,
                                    sizeof(model_path), error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 2;
    }
    fprintf(stderr, "DEEPSEEK_V4_BACKEND backend=%s gguf=%s\n",
            deepseek_v4_ds4_backend_name(), model_path);
    DeepSeekV4Ds4Session *session = deepseek_v4_ds4_open(
        model_path, options->max_context, options->use_spec,
        error, sizeof(error));
    if (!session) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    if (options->prompt) {
        DeepSeekV4Ds4Stats stats;
        int generated = deepseek_v4_ds4_generate_user(
            session, options->prompt, options->max_new_tokens,
            emit_token, NULL, &stats, error, sizeof(error));
        if (generated < 0)
            fprintf(stderr, "%s\n", error);
        else
            print_perf(&stats);
        putchar('\n');
        deepseek_v4_ds4_close(session);
        return generated < 0;
    }

    fprintf(stderr,
            "floyd chat [DeepSeek V4] - :reset clears context, :exit quits\n");
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
            if (!deepseek_v4_ds4_reset(session)) {
                fprintf(stderr, "failed to reset DeepSeek V4 context\n");
                status = 1;
                break;
            }
            puts("context cleared");
            continue;
        }
        if (!size) continue;
        fputs("\xe2\x97\x86 ", stdout);
        fflush(stdout);
        DeepSeekV4Ds4Stats stats;
        int generated = deepseek_v4_ds4_generate_user(
            session, line, options->max_new_tokens,
            emit_token, NULL, &stats, error, sizeof(error));
        putchar('\n');
        if (generated < 0) {
            fprintf(stderr, "%s\n", error);
            status = 1;
            break;
        }
        print_perf(&stats);
    }
    free(line);
    deepseek_v4_ds4_close(session);
    return status;
}

typedef struct {
    DeepSeekV4Ds4Session *session;
    const DeepSeekV4ServeOptions *options;
    char *output;
    size_t output_size;
    size_t output_capacity;
} ServeContext;

static int serve_buffer_emit(int token, const char *piece, size_t piece_size,
                             void *user_data) {
    ServeContext *context = user_data;
    (void)token;
    if (piece_size > SIZE_MAX - context->output_size - 1) return 0;
    size_t required = context->output_size + piece_size + 1;
    if (required > context->output_capacity) {
        size_t capacity = context->output_capacity ? context->output_capacity : 256;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2) {
                capacity = required;
                break;
            }
            capacity *= 2;
        }
        char *grown = realloc(context->output, capacity);
        if (!grown) return 0;
        context->output = grown;
        context->output_capacity = capacity;
    }
    memcpy(context->output + context->output_size, piece, piece_size);
    context->output_size += piece_size;
    context->output[context->output_size] = 0;
    return 1;
}

static int serve_generate(
    ServeContext *context,
    const DeepSeekV4Ds4Message *messages, size_t message_count,
    int max_tokens, float temperature, float top_p, int draft,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    DeepSeekV4Ds4RequestConfig config = {
        .max_tokens = max_tokens > 0
            ? max_tokens : context->options->max_new_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .draft = draft,
    };
    DeepSeekV4Ds4Stats stats;
    int generated = deepseek_v4_ds4_generate_messages(
        context->session, messages, message_count, &config,
        sink, sink_data, &stats, error, error_size);
    if (generated < 0) return 0;

    result->prompt_tokens = stats.prompt_tokens + stats.cached_tokens;
    result->cached_tokens = stats.cached_tokens;
    result->completion_tokens = stats.generated_tokens;
    result->prompt_ms = stats.prompt_ms;
    result->decode_ms = stats.decode_ms;
    result->cache_entries = stats.cache_entries;
    result->cache_bytes = stats.cache_bytes;
    result->finish_reason = generated >= config.max_tokens ? "length" : "stop";
    return 1;
}

static int serve_request(void *user_data,
                         const DeepSeekV4ServeRequest *request,
                         DeepSeekV4ServeResponse *response,
                         char *error, size_t error_size) {
    ServeContext *context = user_data;
    context->output_size = 0;
    if (context->output) context->output[0] = 0;

    size_t message_count = request->prompt ? (request->system ? 2u : 1u)
                                            : request->message_count;
    DeepSeekV4Ds4Message *messages = calloc(message_count, sizeof(*messages));
    if (!messages) {
        snprintf(error, error_size, "out of memory preparing messages");
        return 0;
    }
    if (request->prompt) {
        size_t index = 0;
        if (request->system) {
            messages[index].role = "system";
            messages[index++].content = request->system;
        }
        messages[index].role = "user";
        messages[index].content = request->prompt;
    } else {
        for (size_t index = 0; index < message_count; ++index) {
            messages[index].role = request->messages[index].role;
            messages[index].content = request->messages[index].content;
        }
    }

    int draft = request->draft >= 0 ? request->draft : context->options->draft;
    if (request->draft < 0 && request->temperature > 0.0f) draft = 1;
    OpenAIGenerationResult result = {0};
    int generated = serve_generate(
        context, messages, message_count, request->max_tokens,
        request->temperature, request->top_p, draft,
        serve_buffer_emit, context, &result, error, error_size);
    free(messages);
    if (!generated) return 0;

    response->text = context->output ? context->output : "";
    response->prompt_tokens = result.prompt_tokens;
    response->cached_tokens = result.cached_tokens;
    response->completion_tokens = result.completion_tokens;
    response->prompt_ms = result.prompt_ms;
    response->decode_ms = result.decode_ms;
    response->cache_hit = result.cached_tokens > 0;
    response->cache_prefix_tokens = result.cached_tokens;
    response->cache_entries = result.cache_entries;
    response->cache_bytes = result.cache_bytes;
    return 1;
}

static int serve_openai_request(
    void *user_data, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size) {
    ServeContext *context = user_data;
    DeepSeekV4Ds4Message *messages = calloc(
        request->message_count, sizeof(*messages));
    if (!messages) {
        snprintf(error, error_size, "out of memory preparing messages");
        return 0;
    }
    for (size_t index = 0; index < request->message_count; ++index) {
        messages[index].role = request->messages[index].role;
        messages[index].content = request->messages[index].content;
    }
    int generated = serve_generate(
        context, messages, request->message_count, request->max_tokens,
        request->temperature, request->top_p, context->options->draft,
        sink, sink_data, result, error, error_size);
    free(messages);
    return generated;
}

int deepseek_v4_serve_run(const DeepSeekV4ServeOptions *options) {
    if (!options || !options->model_dir || options->max_context <= 0 ||
        options->max_new_tokens <= 0 || options->draft < 0 ||
        options->draft > 16 ||
        (!options->stdio &&
         (!options->host || !options->host[0] || options->port < 1 ||
          options->port > 65535 || !options->served_model_name ||
          !options->served_model_name[0]))) {
        fprintf(stderr, "invalid DeepSeek V4 serve options\n");
        return 2;
    }
    char model_path[4096], error[4096] = {0};
    if (!deepseek_v4_ds4_find_model(options->model_dir, model_path,
                                    sizeof(model_path), error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 2;
    }
    if (options->stdio) {
        fprintf(stderr,
                "DEEPSEEK_V4_SERVE transport=stdio backend=%s "
                "prefix_cache_mb=%llu gguf=%s\n",
                deepseek_v4_ds4_backend_name(),
                (unsigned long long)(options->prefix_cache_bytes / (1024 * 1024)),
                model_path);
    } else {
        fprintf(stderr,
                "DEEPSEEK_V4_SERVE transport=http backend=%s listen=%s:%d "
                "model=%s prefix_cache_mb=%llu auth=%s\n",
                deepseek_v4_ds4_backend_name(), options->host, options->port,
                options->served_model_name,
                (unsigned long long)(options->prefix_cache_bytes / (1024 * 1024)),
                options->api_key ? "on" : "off");
    }
    DeepSeekV4Ds4Session *session = deepseek_v4_ds4_open_cached(
        model_path, options->max_context, options->draft > 1,
        options->prefix_cache_bytes, error, sizeof(error));
    if (!session) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    ServeContext context = {
        .session = session,
        .options = options,
    };
    int status;
    if (options->stdio) {
        status = deepseek_v4_serve_stdio(
            stdin, stdout, serve_request, &context);
    } else {
        OpenAIHttpConfig config = {
            .host = options->host,
            .port = options->port,
            .api_key = options->api_key,
            .model_name = options->served_model_name,
        };
        int served = openai_http_serve(
            &config, serve_openai_request, &context, error, sizeof(error));
        if (!served && error[0]) fprintf(stderr, "%s\n", error);
        status = served ? 0 : 1;
    }
    free(context.output);
    deepseek_v4_ds4_close(session);
    return status;
}
