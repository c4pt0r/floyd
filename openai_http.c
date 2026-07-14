#define _GNU_SOURCE
#include "openai_http.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static void openai_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static void openai_field_error(char *error, size_t error_size,
                               const char *field, const char *message) {
    if (error && error_size)
        snprintf(error, error_size, "%s %s", field, message);
}

static char *openai_strdup(const char *value) {
    if (!value) return NULL;
    size_t size = strlen(value) + 1;
    char *copy = malloc(size);
    if (copy) memcpy(copy, value, size);
    return copy;
}

static int openai_bool_field(jval *root, const char *name, int missing,
                             int *out, char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value) {
        *out = missing;
        return 1;
    }
    if (value->t != J_BOOL) {
        openai_field_error(error, error_size, name, "must be a boolean");
        return 0;
    }
    *out = value->boolean;
    return 1;
}

static int openai_token_limit(jval *root, int *out,
                              char *error, size_t error_size) {
    jval *max_tokens = json_get(root, "max_tokens");
    jval *max_completion_tokens = json_get(root, "max_completion_tokens");
    if (max_tokens && max_completion_tokens) {
        openai_error(error, error_size,
                     "max_tokens and max_completion_tokens cannot both be set");
        return 0;
    }
    jval *value = max_tokens ? max_tokens : max_completion_tokens;
    const char *name = max_tokens ? "max_tokens" : "max_completion_tokens";
    if (!value) {
        *out = -1;
        return 1;
    }
    if (value->t != J_NUM || !isfinite(value->num) ||
        floor(value->num) != value->num || value->num < 1.0 ||
        value->num > INT_MAX) {
        openai_field_error(error, error_size, name,
                           "must be an integer in 1..INT_MAX");
        return 0;
    }
    *out = (int)value->num;
    return 1;
}

static int openai_float_field(jval *root, const char *name, float missing,
                              double minimum, double maximum, int minimum_open,
                              float *out, char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value) {
        *out = missing;
        return 1;
    }
    if (value->t != J_NUM || !isfinite(value->num) ||
        (minimum_open ? value->num <= minimum : value->num < minimum) ||
        value->num > maximum) {
        openai_field_error(error, error_size, name,
                           "is outside its supported range");
        return 0;
    }
    float converted = (float)value->num;
    if (!isfinite(converted) || (minimum_open && converted <= minimum)) {
        openai_field_error(error, error_size, name,
                           "is outside its supported range");
        return 0;
    }
    *out = converted;
    return 1;
}

static int openai_reject_field(jval *root, const char *name,
                               char *error, size_t error_size) {
    if (!json_get(root, name)) return 1;
    openai_field_error(error, error_size, name, "is not supported");
    return 0;
}

static int openai_body_has_decoded_nul(const char *body) {
    int in_string = 0;
    for (const char *p = body; *p; p++) {
        if (!in_string) {
            if (*p == '"') in_string = 1;
            continue;
        }
        if (*p == '"') {
            in_string = 0;
            continue;
        }
        if (*p != '\\') continue;
        p++;
        if (!*p) break;
        if (*p == 'u' && p[1] == '0' && p[2] == '0' &&
            p[3] == '0' && p[4] == '0') return 1;
    }
    return 0;
}

static int openai_compare_keys(const void *left, const void *right) {
    const char *const *left_key = left;
    const char *const *right_key = right;
    return strcmp(*left_key, *right_key);
}

static int openai_unique_object_keys(jval *object,
                                     char *error, size_t error_size) {
    if (!object || object->t != J_OBJ || object->len < 2) return 1;
    char *local_keys[32];
    char **keys = local_keys;
    size_t count = (size_t)object->len;
    if (count > sizeof(local_keys) / sizeof(local_keys[0])) {
        if (count > SIZE_MAX / sizeof(*keys)) {
            openai_error(error, error_size, "object has too many keys");
            return 0;
        }
        keys = malloc(count * sizeof(*keys));
        if (!keys) {
            openai_error(error, error_size,
                         "out of memory validating object keys");
            return 0;
        }
    }
    memcpy(keys, object->keys, count * sizeof(*keys));
    qsort(keys, count, sizeof(*keys), openai_compare_keys);
    for (size_t i = 1; i < count; i++) {
        if (strcmp(keys[i - 1], keys[i]) != 0) continue;
        openai_field_error(error, error_size, keys[i], "must not be repeated");
        if (keys != local_keys) free(keys);
        return 0;
    }
    if (keys != local_keys) free(keys);
    return 1;
}

void openai_chat_request_free(OpenAIChatRequest *request) {
    if (!request) return;
    free(request->model);
    for (size_t i = 0; i < request->message_count; i++) {
        free(request->messages[i].role);
        free(request->messages[i].content);
    }
    free(request->messages);
    memset(request, 0, sizeof(*request));
}

int openai_chat_request_parse(
    const char *body, const char *served_model, OpenAIChatRequest *request,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!request) {
        openai_error(error, error_size, "request output is required");
        return 0;
    }
    memset(request, 0, sizeof(*request));
    request->max_tokens = -1;
    request->temperature = 0.0f;
    request->top_p = 1.0f;
    if (!body || !served_model || !served_model[0]) {
        openai_error(error, error_size, "body and served model are required");
        return 0;
    }

    char json_error[256] = {0};
    jval *root = json_parse_full(body, json_error, sizeof(json_error));
    if (!root) {
        openai_error(error, error_size, json_error);
        return 0;
    }
    if (openai_body_has_decoded_nul(body)) {
        openai_error(error, error_size,
                     "JSON strings must not contain escaped U+0000");
        goto fail;
    }
    if (root->t != J_OBJ) {
        openai_error(error, error_size, "request must be a JSON object");
        goto fail;
    }
    if (!openai_unique_object_keys(root, error, error_size)) goto fail;

    static const char *unsupported[] = {
        "audio", "frequency_penalty", "function_call", "functions",
        "logit_bias", "logprobs", "modalities", "parallel_tool_calls",
        "prediction", "presence_penalty", "reasoning_effort",
        "response_format", "seed", "stop", "tool_choice", "tools",
        "top_logprobs",
    };
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++) {
        if (!openai_reject_field(root, unsupported[i], error, error_size))
            goto fail;
    }

    jval *model = json_get(root, "model");
    if (!model || model->t != J_STR || !model->str[0]) {
        openai_error(error, error_size, "model must be a non-empty string");
        goto fail;
    }
    if (strcmp(model->str, served_model) != 0) {
        openai_error(error, error_size,
                     "model does not match the served model");
        goto fail;
    }
    request->model = openai_strdup(model->str);
    if (!request->model) goto oom;

    jval *messages = json_get(root, "messages");
    if (!messages || messages->t != J_ARR || messages->len <= 0) {
        openai_error(error, error_size, "messages must be a non-empty array");
        goto fail;
    }
    request->messages = calloc((size_t)messages->len, sizeof(*request->messages));
    if (!request->messages) goto oom;
    request->message_count = (size_t)messages->len;
    for (int i = 0; i < messages->len; i++) {
        jval *item = messages->kids[i];
        if (!item || item->t != J_OBJ ||
            !openai_unique_object_keys(item, error, error_size)) {
            if (!item || item->t != J_OBJ)
                openai_error(error, error_size, "each message must be an object");
            goto fail;
        }
        jval *role = json_get(item, "role");
        jval *content = json_get(item, "content");
        if (!role || role->t != J_STR ||
            (strcmp(role->str, "system") != 0 &&
             strcmp(role->str, "user") != 0 &&
             strcmp(role->str, "assistant") != 0)) {
            openai_error(error, error_size,
                         "message role must be system, user, or assistant");
            goto fail;
        }
        if (!content || content->t != J_STR) {
            openai_error(error, error_size, "message content must be a string");
            goto fail;
        }
        if (json_get(item, "tool_calls") || json_get(item, "function_call") ||
            json_get(item, "name")) {
            openai_error(error, error_size,
                         "message tool and name fields are not supported");
            goto fail;
        }
        request->messages[i].role = openai_strdup(role->str);
        request->messages[i].content = openai_strdup(content->str);
        if (!request->messages[i].role || !request->messages[i].content) goto oom;
    }

    if (!openai_token_limit(root, &request->max_tokens, error, error_size) ||
        !openai_float_field(root, "temperature", 0.0f, 0.0, 2.0, 0,
                            &request->temperature, error, error_size) ||
        !openai_float_field(root, "top_p", 1.0f, 0.0, 1.0, 1,
                            &request->top_p, error, error_size) ||
        !openai_bool_field(root, "stream", 0, &request->stream,
                           error, error_size)) goto fail;

    jval *n = json_get(root, "n");
    if (n && (n->t != J_NUM || !isfinite(n->num) || n->num != 1.0)) {
        openai_error(error, error_size, "n must be 1");
        goto fail;
    }

    jval *stream_options = json_get(root, "stream_options");
    if (stream_options) {
        if (!request->stream) {
            openai_error(error, error_size,
                         "stream_options requires stream to be true");
            goto fail;
        }
        if (stream_options->t != J_OBJ ||
            !openai_unique_object_keys(stream_options, error, error_size)) {
            if (stream_options->t != J_OBJ)
                openai_error(error, error_size, "stream_options must be an object");
            goto fail;
        }
        for (int i = 0; i < stream_options->len; i++) {
            if (strcmp(stream_options->keys[i], "include_usage") != 0) {
                openai_field_error(error, error_size,
                                   stream_options->keys[i],
                                   "is not a supported stream option");
                goto fail;
            }
        }
        if (!openai_bool_field(stream_options, "include_usage", 0,
                               &request->include_usage,
                               error, error_size)) goto fail;
    }

    json_free(root);
    return 1;

oom:
    openai_error(error, error_size, "out of memory parsing request");
fail:
    json_free(root);
    openai_chat_request_free(request);
    return 0;
}

static int openai_json_string_n(FILE *output, const char *value, size_t size) {
    if (!output || (!value && size)) return 0;
    if (fputc('"', output) == EOF) return 0;
    for (size_t i = 0; i < size; i++) {
        unsigned char c = (unsigned char)value[i];
        switch (c) {
            case '"': if (fputs("\\\"", output) == EOF) return 0; break;
            case '\\': if (fputs("\\\\", output) == EOF) return 0; break;
            case '\b': if (fputs("\\b", output) == EOF) return 0; break;
            case '\f': if (fputs("\\f", output) == EOF) return 0; break;
            case '\n': if (fputs("\\n", output) == EOF) return 0; break;
            case '\r': if (fputs("\\r", output) == EOF) return 0; break;
            case '\t': if (fputs("\\t", output) == EOF) return 0; break;
            default:
                if (c < 0x20) {
                    if (fprintf(output, "\\u%04x", (unsigned)c) < 0) return 0;
                } else if (fputc(c, output) == EOF) {
                    return 0;
                }
                break;
        }
    }
    return fputc('"', output) != EOF;
}

static int openai_json_string(FILE *output, const char *value) {
    return openai_json_string_n(output, value ? value : "",
                                value ? strlen(value) : 0);
}

static FILE *openai_json_stream(char **json, size_t *json_size) {
    if (!json || !json_size) return NULL;
    *json = NULL;
    *json_size = 0;
    return open_memstream(json, json_size);
}

static int openai_json_finish(FILE *output, char **json) {
    if (!output) return 0;
    if (fclose(output) == 0) return 1;
    free(*json);
    *json = NULL;
    return 0;
}

int openai_format_models_json(
    const char *model, char **json, size_t *json_size) {
    if (!model || !json || !json_size) return 0;
    FILE *output = openai_json_stream(json, json_size);
    if (!output) return 0;
    int ok = fputs("{\"object\":\"list\",\"data\":[{\"id\":", output) != EOF &&
             openai_json_string(output, model) &&
             fputs(",\"object\":\"model\",\"created\":0,"
                   "\"owned_by\":\"floyd\"}]}", output) != EOF;
    if (!openai_json_finish(output, json)) return 0;
    if (!ok) {
        free(*json);
        *json = NULL;
        *json_size = 0;
        return 0;
    }
    return 1;
}

int openai_format_completion_json(
    const char *id, const char *model, const char *content,
    const OpenAIGenerationResult *result, char **json, size_t *json_size) {
    if (!id || !model || !content || !result || !json || !json_size) return 0;
    FILE *output = openai_json_stream(json, json_size);
    if (!output) return 0;
    const char *finish_reason = result->finish_reason ? result->finish_reason : "stop";
    long long total_tokens =
        (long long)result->prompt_tokens + result->completion_tokens;
    int ok = fputs("{\"id\":", output) != EOF &&
             openai_json_string(output, id) &&
             fputs(",\"object\":\"chat.completion\",\"created\":0,\"model\":",
                   output) != EOF &&
             openai_json_string(output, model) &&
             fputs(",\"choices\":[{\"index\":0,\"message\":{"
                   "\"role\":\"assistant\",\"content\":", output) != EOF &&
             openai_json_string(output, content) &&
             fputs("},\"finish_reason\":", output) != EOF &&
             openai_json_string(output, finish_reason) &&
             fprintf(output,
                     "}],\"usage\":{\"prompt_tokens\":%d,"
                     "\"completion_tokens\":%d,\"total_tokens\":%lld,"
                     "\"prompt_tokens_details\":{\"cached_tokens\":%d}},"
                     "\"floyd\":{\"prompt_ms\":%.3f,\"decode_ms\":%.3f,"
                     "\"cache_entries\":%llu,\"cache_bytes\":%llu}}",
                     result->prompt_tokens, result->completion_tokens,
                     total_tokens, result->cached_tokens,
                     result->prompt_ms, result->decode_ms,
                     (unsigned long long)result->cache_entries,
                     (unsigned long long)result->cache_bytes) >= 0;
    if (!openai_json_finish(output, json)) return 0;
    if (!ok) {
        free(*json);
        *json = NULL;
        *json_size = 0;
        return 0;
    }
    return 1;
}

static int openai_json_nullable_string(FILE *output, const char *value) {
    return value ? openai_json_string(output, value)
                 : fputs("null", output) != EOF;
}

int openai_format_error_json(
    const char *message, const char *type, const char *param, const char *code,
    char **json, size_t *json_size) {
    if (!message || !type || !json || !json_size) return 0;
    FILE *output = openai_json_stream(json, json_size);
    if (!output) return 0;
    int ok = fputs("{\"error\":{\"message\":", output) != EOF &&
             openai_json_string(output, message) &&
             fputs(",\"type\":", output) != EOF &&
             openai_json_string(output, type) &&
             fputs(",\"param\":", output) != EOF &&
             openai_json_nullable_string(output, param) &&
             fputs(",\"code\":", output) != EOF &&
             openai_json_nullable_string(output, code) &&
             fputs("}}", output) != EOF;
    if (!openai_json_finish(output, json)) return 0;
    if (!ok) {
        free(*json);
        *json = NULL;
        *json_size = 0;
        return 0;
    }
    return 1;
}
