#define _GNU_SOURCE
#include "openai_http.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

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

#define OPENAI_JSON_MAX_DEPTH 128u
#define OPENAI_JSON_MAX_NODES 1048576u

typedef struct {
    const unsigned char *cursor;
    const unsigned char *end;
    size_t nodes;
    const char *error;
} OpenAIStrictJson;

static void openai_strict_json_skip_ws(OpenAIStrictJson *parser) {
    while (parser->cursor < parser->end &&
           (*parser->cursor == ' ' || *parser->cursor == '\t' ||
            *parser->cursor == '\r' || *parser->cursor == '\n'))
        parser->cursor++;
}

static int openai_strict_json_hex4(
    const unsigned char *value, unsigned *codepoint) {
    unsigned result = 0;
    for (int i = 0; i < 4; i++) {
        unsigned char c = value[i];
        unsigned digit;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else return 0;
        result = (result << 4) | digit;
    }
    *codepoint = result;
    return 1;
}

static int openai_strict_json_utf8(
    OpenAIStrictJson *parser, unsigned char lead) {
    size_t size;
    unsigned char second_min = 0x80, second_max = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
        size = 2;
    } else if (lead >= 0xe0 && lead <= 0xef) {
        size = 3;
        if (lead == 0xe0) second_min = 0xa0;
        if (lead == 0xed) second_max = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
        size = 4;
        if (lead == 0xf0) second_min = 0x90;
        if (lead == 0xf4) second_max = 0x8f;
    } else {
        parser->error = "invalid UTF-8 lead byte in JSON string";
        return 0;
    }
    if ((size_t)(parser->end - parser->cursor) < size - 1) {
        parser->error = "incomplete UTF-8 sequence in JSON string";
        return 0;
    }
    unsigned char second = *parser->cursor++;
    if (second < second_min || second > second_max) {
        parser->error = "invalid UTF-8 continuation in JSON string";
        return 0;
    }
    for (size_t i = 2; i < size; i++) {
        unsigned char continuation = *parser->cursor++;
        if (continuation < 0x80 || continuation > 0xbf) {
            parser->error = "invalid UTF-8 continuation in JSON string";
            return 0;
        }
    }
    return 1;
}

static int openai_strict_json_string(OpenAIStrictJson *parser) {
    if (parser->cursor >= parser->end || *parser->cursor++ != '"') {
        parser->error = "expected JSON string";
        return 0;
    }
    while (parser->cursor < parser->end) {
        unsigned char c = *parser->cursor++;
        if (c == '"') return 1;
        if (c < 0x20) {
            parser->error = "unescaped control byte in JSON string";
            return 0;
        }
        if (c >= 0x80) {
            if (!openai_strict_json_utf8(parser, c)) return 0;
            continue;
        }
        if (c != '\\') continue;
        if (parser->cursor >= parser->end) {
            parser->error = "incomplete JSON string escape";
            return 0;
        }
        unsigned char escape = *parser->cursor++;
        if (escape == '"' || escape == '\\' || escape == '/' ||
            escape == 'b' || escape == 'f' || escape == 'n' ||
            escape == 'r' || escape == 't') continue;
        if (escape != 'u' || (size_t)(parser->end - parser->cursor) < 4) {
            parser->error = "invalid JSON string escape";
            return 0;
        }
        unsigned codepoint;
        if (!openai_strict_json_hex4(parser->cursor, &codepoint)) {
            parser->error = "invalid JSON unicode escape";
            return 0;
        }
        parser->cursor += 4;
        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
            if ((size_t)(parser->end - parser->cursor) < 6 ||
                parser->cursor[0] != '\\' || parser->cursor[1] != 'u') {
                parser->error = "missing low JSON unicode surrogate";
                return 0;
            }
            unsigned low;
            if (!openai_strict_json_hex4(parser->cursor + 2, &low) ||
                low < 0xdc00 || low > 0xdfff) {
                parser->error = "invalid low JSON unicode surrogate";
                return 0;
            }
            parser->cursor += 6;
        } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
            parser->error = "unexpected low JSON unicode surrogate";
            return 0;
        }
    }
    parser->error = "unterminated JSON string";
    return 0;
}

static int openai_strict_json_value(OpenAIStrictJson *parser, size_t depth);

static int openai_strict_json_container(
    OpenAIStrictJson *parser, size_t depth, int object) {
    if (depth >= OPENAI_JSON_MAX_DEPTH) {
        parser->error = "JSON nesting depth exceeds 128";
        return 0;
    }
    unsigned char end = object ? '}' : ']';
    parser->cursor++;
    openai_strict_json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == end) {
        parser->cursor++;
        return 1;
    }
    for (;;) {
        if (object) {
            if (!openai_strict_json_string(parser)) return 0;
            openai_strict_json_skip_ws(parser);
            if (parser->cursor >= parser->end || *parser->cursor++ != ':') {
                parser->error = "expected colon after JSON object key";
                return 0;
            }
        }
        if (!openai_strict_json_value(parser, depth + 1)) return 0;
        openai_strict_json_skip_ws(parser);
        if (parser->cursor >= parser->end) {
            parser->error = "unterminated JSON container";
            return 0;
        }
        unsigned char delimiter = *parser->cursor++;
        if (delimiter == end) return 1;
        if (delimiter != ',') {
            parser->error = "expected comma or JSON container end";
            return 0;
        }
        openai_strict_json_skip_ws(parser);
    }
}

static int openai_strict_json_number(OpenAIStrictJson *parser) {
    const unsigned char *start = parser->cursor;
    if (parser->cursor < parser->end && *parser->cursor == '-')
        parser->cursor++;
    if (parser->cursor >= parser->end) goto invalid;
    if (*parser->cursor == '0') {
        parser->cursor++;
        if (parser->cursor < parser->end &&
            *parser->cursor >= '0' && *parser->cursor <= '9') goto invalid;
    } else if (*parser->cursor >= '1' && *parser->cursor <= '9') {
        do parser->cursor++;
        while (parser->cursor < parser->end &&
               *parser->cursor >= '0' && *parser->cursor <= '9');
    } else {
        goto invalid;
    }
    if (parser->cursor < parser->end && *parser->cursor == '.') {
        parser->cursor++;
        if (parser->cursor >= parser->end ||
            *parser->cursor < '0' || *parser->cursor > '9') goto invalid;
        do parser->cursor++;
        while (parser->cursor < parser->end &&
               *parser->cursor >= '0' && *parser->cursor <= '9');
    }
    if (parser->cursor < parser->end &&
        (*parser->cursor == 'e' || *parser->cursor == 'E')) {
        parser->cursor++;
        if (parser->cursor < parser->end &&
            (*parser->cursor == '+' || *parser->cursor == '-'))
            parser->cursor++;
        if (parser->cursor >= parser->end ||
            *parser->cursor < '0' || *parser->cursor > '9') goto invalid;
        do parser->cursor++;
        while (parser->cursor < parser->end &&
               *parser->cursor >= '0' && *parser->cursor <= '9');
    }
    return parser->cursor > start;

invalid:
    parser->error = "invalid JSON number";
    return 0;
}

static int openai_strict_json_literal(
    OpenAIStrictJson *parser, const char *literal, size_t size) {
    if ((size_t)(parser->end - parser->cursor) < size ||
        memcmp(parser->cursor, literal, size) != 0) {
        parser->error = "invalid JSON literal";
        return 0;
    }
    parser->cursor += size;
    return 1;
}

static int openai_strict_json_value(OpenAIStrictJson *parser, size_t depth) {
    openai_strict_json_skip_ws(parser);
    if (++parser->nodes > OPENAI_JSON_MAX_NODES) {
        parser->error = "JSON node limit exceeds 1048576";
        return 0;
    }
    if (parser->cursor >= parser->end) {
        parser->error = "expected JSON value";
        return 0;
    }
    switch (*parser->cursor) {
        case '{': return openai_strict_json_container(parser, depth, 1);
        case '[': return openai_strict_json_container(parser, depth, 0);
        case '"': return openai_strict_json_string(parser);
        case 't': return openai_strict_json_literal(parser, "true", 4);
        case 'f': return openai_strict_json_literal(parser, "false", 5);
        case 'n': return openai_strict_json_literal(parser, "null", 4);
        default: return openai_strict_json_number(parser);
    }
}

static int openai_strict_json_validate(
    const char *body, char *error, size_t error_size) {
    OpenAIStrictJson parser = {
        .cursor = (const unsigned char *)body,
        .end = (const unsigned char *)body + strlen(body),
    };
    int valid = openai_strict_json_value(&parser, 0);
    if (valid) {
        openai_strict_json_skip_ws(&parser);
        if (parser.cursor != parser.end) {
            parser.error = "trailing data after JSON value";
            valid = 0;
        }
    }
    if (!valid && error && error_size)
        snprintf(error, error_size, "invalid JSON: %s",
                 parser.error ? parser.error : "invalid value");
    return valid;
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

static int openai_chat_request_parse_detailed(
    const char *body, const char *served_model, OpenAIChatRequest *request,
    char *error, size_t error_size,
    char *error_param, size_t error_param_size) {
    if (error && error_size) error[0] = 0;
    if (error_param && error_param_size) error_param[0] = 0;
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

    if (!openai_strict_json_validate(body, error, error_size)) return 0;

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

    static const char *allowed[] = {
        "model", "messages", "max_tokens", "max_completion_tokens",
        "temperature", "top_p", "stream", "stream_options", "n",
    };
    for (int i = 0; i < root->len; i++) {
        int accepted = 0;
        for (size_t j = 0; j < sizeof(allowed) / sizeof(allowed[0]); j++)
            if (strcmp(root->keys[i], allowed[j]) == 0) {
                accepted = 1;
                break;
            }
        if (!accepted) {
            openai_field_error(
                error, error_size, root->keys[i], "is not supported");
            if (error_param && error_param_size)
                snprintf(error_param, error_param_size, "%s", root->keys[i]);
            goto fail;
        }
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
        for (int field = 0; field < item->len; field++) {
            if (strcmp(item->keys[field], "role") != 0 &&
                strcmp(item->keys[field], "content") != 0) {
                openai_field_error(
                    error, error_size, item->keys[field],
                    "is not supported in a message");
                goto fail;
            }
        }
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
                if (error_param && error_param_size)
                    snprintf(error_param, error_param_size, "stream_options.%s",
                             stream_options->keys[i]);
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

int openai_chat_request_parse(
    const char *body, const char *served_model, OpenAIChatRequest *request,
    char *error, size_t error_size) {
    return openai_chat_request_parse_detailed(
        body, served_model, request, error, error_size, NULL, 0);
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

static long long openai_created_now(void) {
    time_t created = time(NULL);
    return created == (time_t)-1 || created <= 0 ? 1 : (long long)created;
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
    const char *id, const char *model, const char *content, size_t content_size,
    const OpenAIGenerationResult *result, char **json, size_t *json_size) {
    if (!id || !model || (!content && content_size) ||
        !result || !json || !json_size) return 0;
    FILE *output = openai_json_stream(json, json_size);
    if (!output) return 0;
    const char *finish_reason = result->finish_reason ? result->finish_reason : "stop";
    long long total_tokens =
        (long long)result->prompt_tokens + result->completion_tokens;
    long long created = openai_created_now();
    int ok = fputs("{\"id\":", output) != EOF &&
             openai_json_string(output, id) &&
             fputs(",\"object\":\"chat.completion\",\"created\":", output) != EOF &&
             fprintf(output, "%lld,\"model\":", created) >= 0 &&
             openai_json_string(output, model) &&
             fputs(",\"choices\":[{\"index\":0,\"message\":{"
                   "\"role\":\"assistant\",\"content\":", output) != EOF &&
             openai_json_string_n(output, content, content_size) &&
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

#define OPENAI_HTTP_HEADER_LIMIT (32u * 1024u)
#define OPENAI_HTTP_BODY_LIMIT (8u * 1024u * 1024u)

typedef struct {
    char method[16];
    char path[2048];
    int has_content_length;
    size_t content_length;
    int has_transfer_encoding;
    int authorization_count;
    int authorization_invalid_ows;
    char *authorization;
    int host_count;
    int host_nonempty;
} OpenAIHttpRequest;

typedef struct {
    int fd;
    int wake_fd;
    int timeout_ms;
} OpenAIHttpConnection;

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
    int failed;
} OpenAIContentBuffer;

typedef struct {
    OpenAIHttpConnection *connection;
    OpenAIContentBuffer *buffer;
} OpenAIAccumulationSink;

typedef struct {
    OpenAIHttpConnection *connection;
    const char *id;
    const char *model;
    long long created;
    int committed;
    int failed;
} OpenAISseSink;

typedef struct {
    OpenAITokenSink sink;
    void *sink_data;
    unsigned char pending[4];
    size_t pending_size;
    OpenAIContentBuffer output;
    int last_token;
    int failed;
} OpenAIUtf8Sink;

static volatile sig_atomic_t openai_http_stop;
static int openai_http_wakeup[2] = {-1, -1};

static void openai_http_drain_wakeup(int fd) {
    char buffer[64];
    while (read(fd, buffer, sizeof(buffer)) > 0) {}
}

static int64_t openai_http_now_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static int openai_http_wait(
    OpenAIHttpConnection *connection, short events, int64_t deadline) {
    struct pollfd fds[2] = {
        {.fd = connection->fd, .events = events},
        {.fd = connection->wake_fd, .events = POLLIN},
    };
    for (;;) {
        if (connection->wake_fd >= 0 && openai_http_stop) return 0;
        int64_t now = openai_http_now_ms();
        if (now < 0 || now >= deadline) return -1;
        int64_t remaining = deadline - now;
        int timeout = remaining > INT_MAX ? INT_MAX : (int)remaining;
        int count = connection->wake_fd >= 0 ? 2 : 1;
        int polled = poll(fds, (nfds_t)count, timeout);
        if (polled < 0) {
            if (errno == EINTR) {
                if (connection->wake_fd >= 0 && openai_http_stop) return 0;
                continue;
            }
            return 0;
        }
        if (polled == 0) return -1;
        if (connection->wake_fd >= 0 && fds[1].revents) {
            openai_http_drain_wakeup(connection->wake_fd);
            if (openai_http_stop) return 0;
            continue;
        }
        if (fds[0].revents & (events | POLLHUP | POLLERR | POLLNVAL))
            return 1;
    }
}

static int openai_http_send_all(
    OpenAIHttpConnection *connection, const void *data, size_t size) {
    const char *bytes = data;
    size_t offset = 0;
    int64_t now = openai_http_now_ms();
    if (now < 0) return 0;
    int64_t deadline = now + connection->timeout_ms;
    while (offset < size) {
        now = openai_http_now_ms();
        if (now < 0 || now >= deadline) return 0;
#ifdef MSG_NOSIGNAL
        ssize_t sent = send(
            connection->fd, bytes + offset, size - offset, MSG_NOSIGNAL);
#else
        ssize_t sent = send(connection->fd, bytes + offset, size - offset, 0);
#endif
        if (sent > 0) {
            offset += (size_t)sent;
            continue;
        }
        if (sent < 0 && errno == EINTR) {
            if (connection->wake_fd >= 0 && openai_http_stop) return 0;
            continue;
        }
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (openai_http_wait(connection, POLLOUT, deadline) == 1) continue;
        }
        return 0;
    }
    return 1;
}

static const char *openai_http_reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

static int openai_http_send_response(
    OpenAIHttpConnection *connection, int status,
    const char *content_type, const char *extra_headers,
    const char *body, size_t body_size) {
    char headers[1024];
    int header_size = snprintf(
        headers, sizeof(headers),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n%s\r\n",
        status, openai_http_reason(status), content_type, body_size,
        extra_headers ? extra_headers : "");
    if (header_size < 0 || (size_t)header_size >= sizeof(headers)) return 0;
    return openai_http_send_all(connection, headers, (size_t)header_size) &&
           openai_http_send_all(connection, body, body_size);
}

static int openai_http_send_error(
    OpenAIHttpConnection *connection, int status, const char *message,
    const char *param, const char *code, const char *extra_headers) {
    char *json = NULL;
    size_t json_size = 0;
    const char *type = status == 401 ? "authentication_error" :
                       status == 500 ? "server_error" :
                                       "invalid_request_error";
    if (!openai_format_error_json(
            message, type, param, code, &json, &json_size)) return 0;
    int ok = openai_http_send_response(
        connection, status, "application/json", extra_headers, json, json_size);
    free(json);
    return ok;
}

static size_t openai_http_header_end(const char *data, size_t size) {
    if (size < 4) return 0;
    for (size_t i = 0; i + 4 <= size; i++) {
        if (data[i] == '\r' && data[i + 1] == '\n' &&
            data[i + 2] == '\r' && data[i + 3] == '\n') return i + 4;
    }
    return 0;
}

static int openai_http_read_head(
    OpenAIHttpConnection *connection, char *data,
    size_t *data_size, size_t *header_size) {
    *data_size = 0;
    *header_size = 0;
    int64_t now = openai_http_now_ms();
    if (now < 0) return 0;
    int64_t deadline = now + connection->timeout_ms;
    for (;;) {
        size_t end = openai_http_header_end(data, *data_size);
        if (end) {
            if (end > OPENAI_HTTP_HEADER_LIMIT) return 431;
            *header_size = end;
            return 1;
        }
        if (*data_size > OPENAI_HTTP_HEADER_LIMIT) return 431;
        int ready = openai_http_wait(connection, POLLIN, deadline);
        if (ready < 0) return 408;
        if (!ready) return 0;
        size_t remaining = OPENAI_HTTP_HEADER_LIMIT + 1 - *data_size;
        ssize_t received = recv(
            connection->fd, data + *data_size, remaining, 0);
        if (received > 0) {
            *data_size += (size_t)received;
            continue;
        }
        if (received < 0 && errno == EINTR) {
            if (connection->wake_fd >= 0 && openai_http_stop) return 0;
            continue;
        }
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            continue;
        if (received == 0)
            return connection->wake_fd >= 0 && openai_http_stop ? 0 : 400;
        return 0;
    }
}

static char *openai_http_trim_value(char *value) {
    while (*value == ' ' || *value == '\t') value++;
    char *end = value + strlen(value);
    while (end > value && (end[-1] == ' ' || end[-1] == '\t')) end--;
    *end = 0;
    return value;
}

static int openai_http_parse_size(const char *value, size_t *out) {
    if (!value || !value[0]) return 0;
    size_t parsed = 0;
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p < '0' || *p > '9') return 0;
        unsigned digit = (unsigned)(*p - '0');
        if (parsed > (SIZE_MAX - digit) / 10) return 0;
        parsed = parsed * 10 + digit;
    }
    *out = parsed;
    return 1;
}

static int openai_http_parse_head(
    const char *data, size_t header_size, OpenAIHttpRequest *request) {
    if (memchr(data, 0, header_size)) return 0;
    char *copy = malloc(header_size + 1);
    if (!copy) return -1;
    memcpy(copy, data, header_size);
    copy[header_size] = 0;
    memset(request, 0, sizeof(*request));

    char *line_end = strstr(copy, "\r\n");
    if (!line_end) goto malformed;
    *line_end = 0;
    char version[16];
    char extra;
    if (sscanf(copy, "%15s %2047s %15s %c",
               request->method, request->path, version, &extra) != 3 ||
        strcmp(version, "HTTP/1.1") != 0) goto malformed;

    char *line = line_end + 2;
    while (line[0]) {
        line_end = strstr(line, "\r\n");
        if (!line_end) goto malformed;
        if (line_end == line) break;
        *line_end = 0;
        char *colon = strchr(line, ':');
        if (!colon || colon == line) goto malformed;
        *colon = 0;
        for (char *p = line; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (c <= 0x20 || c >= 0x7f) goto malformed;
        }
        char *raw_value = colon + 1;
        size_t raw_value_size = strlen(raw_value);
        int trailing_ows = raw_value_size &&
            (raw_value[raw_value_size - 1] == ' ' ||
             raw_value[raw_value_size - 1] == '\t');
        char *value = openai_http_trim_value(raw_value);
        if (strcasecmp(line, "Content-Length") == 0) {
            size_t parsed = 0;
            if (!openai_http_parse_size(value, &parsed)) goto malformed;
            if (request->has_content_length &&
                request->content_length != parsed) goto malformed;
            request->has_content_length = 1;
            request->content_length = parsed;
        } else if (strcasecmp(line, "Transfer-Encoding") == 0) {
            request->has_transfer_encoding = 1;
        } else if (strcasecmp(line, "Authorization") == 0) {
            request->authorization_count++;
            request->authorization_invalid_ows |= trailing_ows ||
                raw_value[0] == '\t' ||
                (raw_value[0] == ' ' &&
                 (raw_value[1] == ' ' || raw_value[1] == '\t'));
            request->authorization = value;
        } else if (strcasecmp(line, "Host") == 0) {
            request->host_count++;
            request->host_nonempty = value[0] != 0;
        }
        line = line_end + 2;
    }

    if (request->authorization) {
        size_t offset = (size_t)(request->authorization - copy);
        size_t size = strlen(request->authorization) + 1;
        request->authorization = malloc(size);
        if (!request->authorization) {
            free(copy);
            return -1;
        }
        memcpy(request->authorization, copy + offset, size);
    }
    free(copy);
    return 1;

malformed:
    free(copy);
    return 0;
}

static void openai_http_request_free(OpenAIHttpRequest *request) {
    free(request->authorization);
    request->authorization = NULL;
}

static int openai_http_authorized(
    const OpenAIHttpConfig *config, const OpenAIHttpRequest *request) {
    if (!config->api_key) return 1;
    if (request->authorization_count != 1 || !request->authorization ||
        request->authorization_invalid_ows) return 0;
    static const char prefix[] = "Bearer ";
    size_t prefix_size = sizeof(prefix) - 1;
    size_t key_size = strlen(config->api_key);
    size_t authorization_size = strlen(request->authorization);
    if (authorization_size != prefix_size + key_size ||
        memcmp(request->authorization, prefix, prefix_size) != 0)
        return 0;
    const unsigned char *provided = (const unsigned char *)
        request->authorization + prefix_size;
    const unsigned char *expected = (const unsigned char *)config->api_key;
    unsigned char difference = 0;
    for (size_t i = 0; i < key_size; i++)
        difference |= provided[i] ^ expected[i];
    return difference == 0;
}

static int openai_http_read_body(
    OpenAIHttpConnection *connection,
    const char *head, size_t data_size, size_t header_size,
    size_t body_size, char **body) {
    *body = malloc(body_size + 1);
    if (!*body) return -1;
    size_t buffered = data_size - header_size;
    if (buffered > body_size) buffered = body_size;
    if (buffered) memcpy(*body, head + header_size, buffered);
    size_t offset = buffered;
    int64_t now = openai_http_now_ms();
    if (now < 0) {
        free(*body);
        *body = NULL;
        return -2;
    }
    int64_t deadline = now + connection->timeout_ms;
    while (offset < body_size) {
        int ready = openai_http_wait(connection, POLLIN, deadline);
        if (ready <= 0) {
            free(*body);
            *body = NULL;
            return ready < 0 ? -3 : -2;
        }
        ssize_t received = recv(
            connection->fd, *body + offset, body_size - offset, 0);
        if (received > 0) {
            offset += (size_t)received;
            continue;
        }
        if (received < 0 && errno == EINTR) {
            if (connection->wake_fd >= 0 && openai_http_stop) {
                free(*body);
                *body = NULL;
                return -2;
            }
            continue;
        }
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            continue;
        if (received == 0) {
            free(*body);
            *body = NULL;
            return connection->wake_fd >= 0 && openai_http_stop ? -2 : 0;
        }
        free(*body);
        *body = NULL;
        return -2;
    }
    (*body)[body_size] = 0;
    return 1;
}

static int openai_content_buffer_append(
    OpenAIContentBuffer *buffer, const char *piece, size_t piece_size) {
    if ((!piece && piece_size) || buffer->failed) return 0;
    if (piece_size > SIZE_MAX - buffer->size - 1) {
        buffer->failed = 1;
        return 0;
    }
    size_t required = buffer->size + piece_size + 1;
    if (required > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 256;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2) {
                capacity = required;
                break;
            }
            capacity *= 2;
        }
        char *grown = realloc(buffer->data, capacity);
        if (!grown) {
            buffer->failed = 1;
            return 0;
        }
        buffer->data = grown;
        buffer->capacity = capacity;
    }
    if (piece_size) memcpy(buffer->data + buffer->size, piece, piece_size);
    buffer->size += piece_size;
    buffer->data[buffer->size] = 0;
    return 1;
}

static int openai_content_sink(
    int token, const char *piece, size_t piece_size, void *user_data) {
    (void)token;
    OpenAIAccumulationSink *sink = user_data;
    struct pollfd peer = {
        .fd = sink->connection->fd,
        .events = POLLIN,
    };
    int status;
    do status = poll(&peer, 1, 0);
    while (status < 0 && errno == EINTR);
    if (status > 0 && (peer.revents & (POLLHUP | POLLERR | POLLNVAL)))
        return 0;
    return openai_content_buffer_append(sink->buffer, piece, piece_size);
}

static int openai_utf8_expected_size(unsigned char lead) {
    if (lead <= 0x7f) return 1;
    if (lead >= 0xc2 && lead <= 0xdf) return 2;
    if (lead >= 0xe0 && lead <= 0xef) return 3;
    if (lead >= 0xf0 && lead <= 0xf4) return 4;
    return 0;
}

static int openai_utf8_continuation_valid(
    const unsigned char *pending, size_t index, unsigned char byte) {
    if (byte < 0x80 || byte > 0xbf) return 0;
    if (index != 1) return 1;
    unsigned char lead = pending[0];
    if (lead == 0xe0 && byte < 0xa0) return 0;
    if (lead == 0xed && byte > 0x9f) return 0;
    if (lead == 0xf0 && byte < 0x90) return 0;
    if (lead == 0xf4 && byte > 0x8f) return 0;
    return 1;
}

static int openai_utf8_append_replacement(OpenAIUtf8Sink *sink) {
    static const char replacement[] = "\xef\xbf\xbd";
    return openai_content_buffer_append(
        &sink->output, replacement, sizeof(replacement) - 1);
}

static int openai_utf8_emit(OpenAIUtf8Sink *sink, int token) {
    if (!sink->output.size) return 1;
    int ok = sink->sink(
        token, sink->output.data, sink->output.size, sink->sink_data);
    sink->output.size = 0;
    if (sink->output.data) sink->output.data[0] = 0;
    if (!ok) sink->failed = 1;
    return ok;
}

static int openai_utf8_token_sink(
    int token, const char *piece, size_t piece_size, void *user_data) {
    OpenAIUtf8Sink *sink = user_data;
    if (!sink || !sink->sink) return 0;
    if (sink->failed || sink->output.failed) return 0;
    if (!piece && piece_size) {
        sink->failed = 1;
        return 0;
    }
    sink->last_token = token;
    const unsigned char *bytes = (const unsigned char *)piece;
    size_t index = 0;
    while (index < piece_size) {
        unsigned char byte = bytes[index++];
        if (!sink->pending_size) {
            int expected = openai_utf8_expected_size(byte);
            if (expected == 1) {
                char value = (char)byte;
                if (!openai_content_buffer_append(&sink->output, &value, 1))
                    goto fail;
            } else if (expected > 1) {
                sink->pending[sink->pending_size++] = byte;
            } else if (!openai_utf8_append_replacement(sink)) {
                goto fail;
            }
            continue;
        }

        if (!openai_utf8_continuation_valid(
                sink->pending, sink->pending_size, byte)) {
            if (!openai_utf8_append_replacement(sink)) goto fail;
            sink->pending_size = 0;
            index--;
            continue;
        }
        sink->pending[sink->pending_size++] = byte;
        int expected = openai_utf8_expected_size(sink->pending[0]);
        if (sink->pending_size == (size_t)expected) {
            if (!openai_content_buffer_append(
                    &sink->output, (const char *)sink->pending,
                    sink->pending_size)) goto fail;
            sink->pending_size = 0;
        }
    }
    return openai_utf8_emit(sink, token);

fail:
    sink->failed = 1;
    return 0;
}

static int openai_utf8_finish(OpenAIUtf8Sink *sink) {
    if (!sink || sink->failed || sink->output.failed) return 0;
    if (sink->pending_size) {
        if (!openai_utf8_append_replacement(sink)) {
            sink->failed = 1;
            return 0;
        }
        sink->pending_size = 0;
    }
    return openai_utf8_emit(sink, sink->last_token);
}

static void openai_utf8_free(OpenAIUtf8Sink *sink) {
    if (!sink) return;
    free(sink->output.data);
    sink->output.data = NULL;
}

static FILE *openai_sse_event_stream(char **event, size_t *event_size) {
    *event = NULL;
    *event_size = 0;
    return open_memstream(event, event_size);
}

static int openai_sse_send_event(
    OpenAIHttpConnection *connection,
    char *event, size_t event_size, int formatted) {
    int ok = formatted &&
             openai_http_send_all(connection, "data: ", 6) &&
             openai_http_send_all(connection, event, event_size) &&
             openai_http_send_all(connection, "\n\n", 2);
    free(event);
    return ok;
}

static int openai_sse_send_role(
    OpenAIHttpConnection *connection, const char *id,
    const char *model, long long created) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    int formatted = fputs("{\"id\":", output) != EOF &&
                    openai_json_string(output, id) &&
                    fputs(",\"object\":\"chat.completion.chunk\","
                          "\"created\":", output) != EOF &&
                    fprintf(output, "%lld,\"model\":", created) >= 0 &&
                    openai_json_string(output, model) &&
                    fputs(",\"choices\":[{\"index\":0,\"delta\":{"
                          "\"role\":\"assistant\"},"
                          "\"finish_reason\":null}]}", output) != EOF;
    if (fclose(output) != 0) formatted = 0;
    return openai_sse_send_event(
        connection, event, event_size, formatted);
}

static int openai_sse_send_content(
    OpenAIHttpConnection *connection, const char *id,
    const char *model, long long created,
    const char *piece, size_t piece_size) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    int formatted = fputs("{\"id\":", output) != EOF &&
                    openai_json_string(output, id) &&
                    fputs(",\"object\":\"chat.completion.chunk\","
                          "\"created\":", output) != EOF &&
                    fprintf(output, "%lld,\"model\":", created) >= 0 &&
                    openai_json_string(output, model) &&
                    fputs(",\"choices\":[{\"index\":0,\"delta\":{"
                          "\"content\":", output) != EOF &&
                    openai_json_string_n(output, piece, piece_size) &&
                    fputs("},\"finish_reason\":null}]}", output) != EOF;
    if (fclose(output) != 0) formatted = 0;
    return openai_sse_send_event(
        connection, event, event_size, formatted);
}

static int openai_http_send_sse_headers(
    OpenAIHttpConnection *connection);

static int openai_sse_token_sink(
    int token, const char *piece, size_t piece_size, void *user_data) {
    (void)token;
    OpenAISseSink *sink = user_data;
    if (sink->failed || (!piece && piece_size)) return 0;
    if (!sink->committed) {
        if (!openai_http_send_sse_headers(sink->connection) ||
            !openai_sse_send_role(
                sink->connection, sink->id, sink->model, sink->created)) {
            sink->failed = 1;
            return 0;
        }
        sink->committed = 1;
    }
    if (!openai_sse_send_content(
            sink->connection, sink->id, sink->model, sink->created,
            piece, piece_size)) {
        sink->failed = 1;
        return 0;
    }
    return 1;
}

static int openai_sse_send_terminal(
    OpenAIHttpConnection *connection, const char *id,
    const char *model, long long created,
    const OpenAIGenerationResult *result) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    const char *reason = result->finish_reason ? result->finish_reason : "stop";
    int formatted = fputs("{\"id\":", output) != EOF &&
                    openai_json_string(output, id) &&
                    fputs(",\"object\":\"chat.completion.chunk\","
                          "\"created\":", output) != EOF &&
                    fprintf(output, "%lld,\"model\":", created) >= 0 &&
                    openai_json_string(output, model) &&
                    fputs(",\"choices\":[{\"index\":0,\"delta\":{},"
                          "\"finish_reason\":", output) != EOF &&
                    openai_json_string(output, reason) &&
                    fputs("}]}", output) != EOF;
    if (fclose(output) != 0) formatted = 0;
    return openai_sse_send_event(
        connection, event, event_size, formatted);
}

static int openai_sse_send_usage(
    OpenAIHttpConnection *connection, const char *id,
    const char *model, long long created,
    const OpenAIGenerationResult *result) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    long long total_tokens =
        (long long)result->prompt_tokens + result->completion_tokens;
    int formatted = fputs("{\"id\":", output) != EOF &&
                    openai_json_string(output, id) &&
                    fputs(",\"object\":\"chat.completion.chunk\","
                          "\"created\":", output) != EOF &&
                    fprintf(output, "%lld,\"model\":", created) >= 0 &&
                    openai_json_string(output, model) &&
                    fprintf(output,
                            ",\"choices\":[],\"usage\":{"
                            "\"prompt_tokens\":%d,"
                            "\"completion_tokens\":%d,"
                            "\"total_tokens\":%lld,"
                            "\"prompt_tokens_details\":{"
                            "\"cached_tokens\":%d}},"
                            "\"floyd\":{\"prompt_ms\":%.3f,"
                            "\"decode_ms\":%.3f,"
                            "\"cache_entries\":%llu,"
                            "\"cache_bytes\":%llu}}",
                            result->prompt_tokens, result->completion_tokens,
                            total_tokens, result->cached_tokens,
                            result->prompt_ms, result->decode_ms,
                            (unsigned long long)result->cache_entries,
                            (unsigned long long)result->cache_bytes) >= 0;
    if (fclose(output) != 0) formatted = 0;
    return openai_sse_send_event(
        connection, event, event_size, formatted);
}

static int openai_http_send_sse_headers(
    OpenAIHttpConnection *connection) {
    static const char headers[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n\r\n";
    return openai_http_send_all(connection, headers, sizeof(headers) - 1);
}

static int openai_http_make_completion_id(char *id, size_t id_size) {
    static uint64_t counter;
    struct timespec now;
    if (!id || id_size == 0 || clock_gettime(CLOCK_REALTIME, &now) != 0)
        return 0;
    uint64_t sequence = __sync_add_and_fetch(&counter, 1);
    int size = snprintf(
        id, id_size, "chatcmpl-%llx-%llx-%llx",
        (unsigned long long)now.tv_sec,
        (unsigned long long)now.tv_nsec,
        (unsigned long long)sequence);
    return size > 0 && (size_t)size < id_size;
}

static int openai_sse_send_error(
    OpenAIHttpConnection *connection, int client_error,
    const char *message) {
    char *json = NULL;
    size_t json_size = 0;
    const char *safe_message = client_error && message && message[0]
        ? message : "generation failed";
    const char *type = client_error
        ? "invalid_request_error" : "server_error";
    if (!openai_format_error_json(
            safe_message, type, NULL, "generation_failed",
            &json, &json_size)) return 0;
    return openai_sse_send_event(connection, json, json_size, 1);
}

static int openai_http_run_completion(
    OpenAIHttpConnection *connection, const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler,
    void *user_data, char *body) {
    OpenAIChatRequest request = {0};
    char request_error[256] = {0};
    char request_param[128] = {0};
    if (!openai_chat_request_parse_detailed(
            body, config->model_name, &request,
            request_error, sizeof(request_error),
            request_param, sizeof(request_param))) {
        return openai_http_send_error(
            connection, 400,
            request_error[0] ? request_error : "invalid request",
            request_param[0] ? request_param : NULL,
            "invalid_request", NULL);
    }
    if (!handler) {
        openai_chat_request_free(&request);
        return openai_http_send_error(
            connection, 500, "generation handler is unavailable",
            NULL, "server_error", NULL);
    }

    OpenAIGenerationResult result = {0};
    char generation_error[256] = {0};
    char id[96];
    if (!openai_http_make_completion_id(id, sizeof(id))) {
        openai_chat_request_free(&request);
        return openai_http_send_error(
            connection, 500, "could not create completion ID",
            NULL, "server_error", NULL);
    }
    if (request.stream) {
        long long created = openai_created_now();
        OpenAISseSink sink = {
            .connection = connection,
            .id = id,
            .model = request.model,
            .created = created,
        };
        OpenAIUtf8Sink utf8 = {
            .sink = openai_sse_token_sink,
            .sink_data = &sink,
        };
        int generated = handler(
            user_data, &request, openai_utf8_token_sink, &utf8,
            &result, generation_error, sizeof(generation_error));
        int finalized = generated == OPENAI_GENERATE_OK &&
                        openai_utf8_finish(&utf8);
        openai_utf8_free(&utf8);
        if (sink.failed) {
            openai_chat_request_free(&request);
            return 0;
        }
        if (generated != OPENAI_GENERATE_OK || !finalized) {
            int client_error = generated == OPENAI_GENERATE_CLIENT_ERROR;
            int ok;
            if (!sink.committed) {
                ok = openai_http_send_error(
                    connection, client_error ? 400 : 500,
                    client_error && generation_error[0]
                        ? generation_error : "generation failed",
                    NULL, "generation_failed", NULL);
            } else {
                ok = openai_sse_send_error(
                    connection, client_error, generation_error) &&
                     openai_http_send_all(
                         connection, "data: [DONE]\n\n", 14);
            }
            openai_chat_request_free(&request);
            return ok;
        }
        int ok = 1;
        if (!sink.committed) {
            ok = openai_http_send_sse_headers(connection) &&
                 openai_sse_send_role(
                     connection, id, request.model, created);
            sink.committed = ok;
        }
        if (ok)
            ok = openai_sse_send_terminal(
                connection, id, request.model, created, &result);
        if (ok && request.include_usage)
            ok = openai_sse_send_usage(
                connection, id, request.model, created, &result);
        if (ok) ok = openai_http_send_all(
            connection, "data: [DONE]\n\n", 14);
        openai_chat_request_free(&request);
        return ok;
    }

    OpenAIContentBuffer content = {0};
    OpenAIAccumulationSink accumulation = {
        .connection = connection,
        .buffer = &content,
    };
    OpenAIUtf8Sink utf8 = {
        .sink = openai_content_sink,
        .sink_data = &accumulation,
    };
    int generated = handler(
        user_data, &request, openai_utf8_token_sink, &utf8,
        &result, generation_error, sizeof(generation_error));
    int finalized = generated == OPENAI_GENERATE_OK &&
                    openai_utf8_finish(&utf8);
    openai_utf8_free(&utf8);
    if (generated != OPENAI_GENERATE_OK || !finalized || content.failed) {
        free(content.data);
        openai_chat_request_free(&request);
        int client_error = generated == OPENAI_GENERATE_CLIENT_ERROR;
        return openai_http_send_error(
            connection, client_error ? 400 : 500,
            client_error && generation_error[0]
                ? generation_error : "generation failed",
            NULL, "generation_failed", NULL);
    }
    if (!content.data) {
        content.data = openai_strdup("");
        if (!content.data) {
            openai_chat_request_free(&request);
            return openai_http_send_error(
                connection, 500, "out of memory formatting response",
                NULL, "server_error", NULL);
        }
    }
    char *json = NULL;
    size_t json_size = 0;
    int formatted = openai_format_completion_json(
        id, request.model, content.data, content.size,
        &result, &json, &json_size);
    free(content.data);
    openai_chat_request_free(&request);
    if (!formatted)
        return openai_http_send_error(
            connection, 500, "could not format response",
            NULL, "server_error", NULL);
    int ok = openai_http_send_response(
        connection, 200, "application/json", NULL, json, json_size);
    free(json);
    return ok;
}

static int openai_http_handle_connection_internal(
    int fd, const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data, int wake_fd) {
    if (fd < 0 || !config || !config->model_name || !config->model_name[0] ||
        config->io_timeout_ms < 0 ||
        (config->api_key && !config->api_key[0]))
        return 0;
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) return 0;
    OpenAIHttpConnection connection = {
        .fd = fd,
        .wake_fd = wake_fd,
        .timeout_ms = config->io_timeout_ms > 0
            ? config->io_timeout_ms : 30000,
    };

    char head[OPENAI_HTTP_HEADER_LIMIT + 1];
    size_t data_size = 0;
    size_t header_size = 0;
    int read_status = openai_http_read_head(
        &connection, head, &data_size, &header_size);
    if (read_status == 431)
        return openai_http_send_error(
            &connection, 431, "request headers exceed 32 KiB",
            NULL, "headers_too_large", NULL);
    if (read_status == 408)
        return openai_http_send_error(
            &connection, 408, "request headers timed out",
            NULL, "request_timeout", NULL);
    if (read_status == 400)
        return openai_http_send_error(
            &connection, 400, "incomplete HTTP headers",
            NULL, "invalid_http", NULL);
    if (read_status != 1) return 0;

    OpenAIHttpRequest request = {0};
    int parsed = openai_http_parse_head(head, header_size, &request);
    if (parsed < 0)
        return openai_http_send_error(
            &connection, 500, "out of memory parsing headers",
            NULL, "server_error", NULL);
    if (!parsed)
        return openai_http_send_error(
            &connection, 400, "malformed HTTP request",
            NULL, "invalid_http", NULL);

    int ok = 0;
    if (request.host_count != 1 || !request.host_nonempty) {
        ok = openai_http_send_error(
            &connection, 400,
            "exactly one non-empty Host header is required",
            "Host", "invalid_host", NULL);
        goto done;
    }
    if (request.has_transfer_encoding) {
        ok = openai_http_send_error(
            &connection, 400, "Transfer-Encoding is not supported",
            "Transfer-Encoding", "unsupported_transfer_encoding", NULL);
        goto done;
    }
    if (request.has_content_length &&
        request.content_length > OPENAI_HTTP_BODY_LIMIT) {
        ok = openai_http_send_error(
            &connection, 413, "request body exceeds 8 MiB",
            NULL, "body_too_large", NULL);
        goto done;
    }
    if (strncmp(request.path, "/v1", 3) == 0 &&
        (request.path[3] == 0 || request.path[3] == '/') &&
        !openai_http_authorized(config, &request)) {
        ok = openai_http_send_error(
            &connection, 401, "invalid API key", NULL, "invalid_api_key",
            "WWW-Authenticate: Bearer\r\n");
        goto done;
    }

    if (strcmp(request.path, "/v1/models") == 0) {
        if (strcmp(request.method, "GET") != 0) {
            ok = openai_http_send_error(
                &connection, 405, "method not allowed",
                NULL, "method_not_allowed",
                "Allow: GET\r\n");
            goto done;
        }
        char *json = NULL;
        size_t json_size = 0;
        if (!openai_format_models_json(config->model_name, &json, &json_size)) {
            ok = openai_http_send_error(
                &connection, 500, "could not format response",
                NULL, "server_error", NULL);
            goto done;
        }
        ok = openai_http_send_response(
            &connection, 200, "application/json", NULL, json, json_size);
        free(json);
        goto done;
    }

    if (strcmp(request.path, "/v1/chat/completions") == 0) {
        if (strcmp(request.method, "POST") != 0) {
            ok = openai_http_send_error(
                &connection, 405, "method not allowed",
                NULL, "method_not_allowed",
                "Allow: POST\r\n");
            goto done;
        }
        if (!request.has_content_length) {
            ok = openai_http_send_error(
                &connection, 400, "Content-Length is required",
                "Content-Length", "missing_content_length", NULL);
            goto done;
        }
        char *body = NULL;
        int body_status = openai_http_read_body(
            &connection, head, data_size, header_size,
            request.content_length, &body);
        if (body_status == 0) {
            ok = openai_http_send_error(
                &connection, 400,
                "request body is shorter than Content-Length",
                NULL, "incomplete_body", NULL);
            goto done;
        }
        if (body_status < 0) {
            if (body_status == -1)
                ok = openai_http_send_error(
                    &connection, 500, "out of memory reading request body",
                    NULL, "server_error", NULL);
            else if (body_status == -3)
                ok = openai_http_send_error(
                    &connection, 408, "request body timed out",
                    NULL, "request_timeout", NULL);
            goto done;
        }
        if (memchr(body, 0, request.content_length)) {
            free(body);
            ok = openai_http_send_error(
                &connection, 400, "request body contains a NUL byte",
                NULL, "invalid_json", NULL);
            goto done;
        }
        ok = openai_http_run_completion(
            &connection, config, handler, user_data, body);
        free(body);
        goto done;
    }

    ok = openai_http_send_error(
        &connection, 404, "route not found", NULL, "not_found", NULL);

done:
    openai_http_request_free(&request);
    return ok;
}

int openai_http_handle_connection(
    int fd, const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data) {
    return openai_http_handle_connection_internal(
        fd, config, handler, user_data, -1);
}

static void openai_http_signal_handler(int signal_number) {
    (void)signal_number;
    int saved_errno = errno;
    openai_http_stop = 1;
    int fd = openai_http_wakeup[1];
    if (fd >= 0) {
        const unsigned char byte = 1;
        (void)write(fd, &byte, 1);
    }
    errno = saved_errno;
}

static int openai_http_configure_pipe(void) {
    if (openai_http_wakeup[0] >= 0) return 1;
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) return 0;
    for (int i = 0; i < 2; i++) {
        int status_flags = fcntl(pipe_fds[i], F_GETFL);
        int descriptor_flags = fcntl(pipe_fds[i], F_GETFD);
        if (status_flags < 0 || descriptor_flags < 0 ||
            fcntl(pipe_fds[i], F_SETFL, status_flags | O_NONBLOCK) != 0 ||
            fcntl(pipe_fds[i], F_SETFD,
                  descriptor_flags | FD_CLOEXEC) != 0) {
            close(pipe_fds[0]);
            close(pipe_fds[1]);
            return 0;
        }
    }
    openai_http_wakeup[0] = pipe_fds[0];
    openai_http_wakeup[1] = pipe_fds[1];
    return 1;
}

int openai_http_serve(
    const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!config || !config->host || !config->host[0] ||
        config->port < 1 || config->port > 65535 ||
        !config->model_name || !config->model_name[0] || !handler ||
        config->io_timeout_ms < 0 ||
        (config->api_key && !config->api_key[0])) {
        openai_error(error, error_size, "invalid HTTP server configuration");
        return 0;
    }
    if (!openai_http_configure_pipe()) {
        openai_error(error, error_size, "could not create HTTP wakeup pipe");
        return 0;
    }
    openai_http_drain_wakeup(openai_http_wakeup[0]);
    openai_http_stop = 0;

    char service[16];
    snprintf(service, sizeof(service), "%d", config->port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo *addresses = NULL;
    int gai_status = getaddrinfo(config->host, service, &hints, &addresses);
    if (gai_status != 0) {
        if (error && error_size)
            snprintf(error, error_size, "could not resolve listen address: %s",
                     gai_strerror(gai_status));
        return 0;
    }

    int listener = -1;
    for (struct addrinfo *address = addresses;
         address; address = address->ai_next) {
        listener = socket(
            address->ai_family, address->ai_socktype, address->ai_protocol);
        if (listener < 0) continue;
        int enabled = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
        if (bind(listener, address->ai_addr, address->ai_addrlen) == 0 &&
            listen(listener, 16) == 0) break;
        close(listener);
        listener = -1;
    }
    freeaddrinfo(addresses);
    if (listener < 0) {
        openai_error(error, error_size, "could not bind HTTP listen socket");
        return 0;
    }

    struct sigaction action;
    struct sigaction old_int;
    struct sigaction old_term;
    memset(&action, 0, sizeof(action));
    action.sa_handler = openai_http_signal_handler;
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGINT);
    sigaddset(&action.sa_mask, SIGTERM);
    if (sigaction(SIGINT, &action, &old_int) != 0) {
        close(listener);
        openai_error(error, error_size, "could not install signal handlers");
        return 0;
    }
    if (sigaction(SIGTERM, &action, &old_term) != 0) {
        sigaction(SIGINT, &old_int, NULL);
        close(listener);
        openai_error(error, error_size, "could not install signal handlers");
        return 0;
    }
    int ok = 1;
    struct pollfd fds[2] = {
        {.fd = listener, .events = POLLIN},
        {.fd = openai_http_wakeup[0], .events = POLLIN},
    };
    while (!openai_http_stop) {
        int polled = poll(fds, 2, -1);
        if (polled < 0) {
            if (errno == EINTR) {
                if (openai_http_stop) break;
                continue;
            }
            openai_error(error, error_size, "could not poll HTTP listener");
            ok = 0;
            break;
        }
        if (fds[1].revents) {
            openai_http_drain_wakeup(openai_http_wakeup[0]);
            if (openai_http_stop) break;
        }
        if (fds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
            openai_error(error, error_size, "HTTP listen socket failed");
            ok = 0;
            break;
        }
        if (!(fds[0].revents & POLLIN)) continue;
        int connection = accept(listener, NULL, NULL);
        if (connection < 0) {
            if (errno == EINTR) {
                if (openai_http_stop) break;
                continue;
            }
            if (openai_http_stop || errno == EBADF) break;
            openai_error(error, error_size, "could not accept HTTP connection");
            ok = 0;
            break;
        }
        openai_http_handle_connection_internal(
            connection, config, handler, user_data,
            openai_http_wakeup[0]);
        close(connection);
    }

    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGTERM, &old_term, NULL);
    close(listener);
    return ok;
}
