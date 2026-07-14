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

#define OPENAI_HTTP_HEADER_LIMIT (32u * 1024u)
#define OPENAI_HTTP_BODY_LIMIT (8u * 1024u * 1024u)
#define OPENAI_HTTP_ID "chatcmpl-floyd"

typedef struct {
    char method[16];
    char path[2048];
    int has_content_length;
    size_t content_length;
    int has_transfer_encoding;
    int authorization_count;
    int authorization_invalid_ows;
    char *authorization;
} OpenAIHttpRequest;

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
    int failed;
} OpenAIContentBuffer;

typedef struct {
    int fd;
    const char *model;
    int failed;
} OpenAISseSink;

static volatile sig_atomic_t openai_http_stop;
static volatile sig_atomic_t openai_http_wakeup_fd = -1;

static void openai_http_drain_wakeup(int fd) {
    char buffer[64];
    while (read(fd, buffer, sizeof(buffer)) > 0) {}
}

static int openai_http_wait_readable(int fd, int wake_fd) {
    if (wake_fd < 0) return 1;
    struct pollfd fds[2] = {
        {.fd = fd, .events = POLLIN},
        {.fd = wake_fd, .events = POLLIN},
    };
    for (;;) {
        if (openai_http_stop) return 0;
        int polled = poll(fds, 2, -1);
        if (polled < 0) {
            if (errno == EINTR) {
                if (openai_http_stop) return 0;
                continue;
            }
            return 0;
        }
        if (fds[1].revents) {
            openai_http_drain_wakeup(wake_fd);
            return 0;
        }
        if (fds[0].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))
            return 1;
    }
}

static int openai_http_send_all(int fd, const void *data, size_t size) {
    const char *bytes = data;
    size_t offset = 0;
    while (offset < size) {
#ifdef MSG_NOSIGNAL
        ssize_t sent = send(fd, bytes + offset, size - offset, MSG_NOSIGNAL);
#else
        ssize_t sent = send(fd, bytes + offset, size - offset, 0);
#endif
        if (sent > 0) {
            offset += (size_t)sent;
            continue;
        }
        if (sent < 0 && errno == EINTR) {
            if (openai_http_stop) return 0;
            continue;
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
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

static int openai_http_send_response(
    int fd, int status, const char *content_type, const char *extra_headers,
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
    return openai_http_send_all(fd, headers, (size_t)header_size) &&
           openai_http_send_all(fd, body, body_size);
}

static int openai_http_send_error(
    int fd, int status, const char *message, const char *code,
    const char *extra_headers) {
    char *json = NULL;
    size_t json_size = 0;
    const char *type = status == 401 ? "authentication_error" :
                       status == 500 ? "server_error" :
                                       "invalid_request_error";
    if (!openai_format_error_json(
            message, type, NULL, code, &json, &json_size)) return 0;
    int ok = openai_http_send_response(
        fd, status, "application/json", extra_headers, json, json_size);
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
    int fd, int wake_fd, char *data,
    size_t *data_size, size_t *header_size) {
    *data_size = 0;
    *header_size = 0;
    for (;;) {
        size_t end = openai_http_header_end(data, *data_size);
        if (end) {
            if (end > OPENAI_HTTP_HEADER_LIMIT) return 431;
            *header_size = end;
            return 1;
        }
        if (*data_size > OPENAI_HTTP_HEADER_LIMIT) return 431;
        if (!openai_http_wait_readable(fd, wake_fd)) return 0;
        size_t remaining = OPENAI_HTTP_HEADER_LIMIT + 1 - *data_size;
        ssize_t received = recv(fd, data + *data_size, remaining, 0);
        if (received > 0) {
            *data_size += (size_t)received;
            continue;
        }
        if (received < 0 && errno == EINTR) {
            if (wake_fd >= 0 && openai_http_stop) return 0;
            continue;
        }
        if (received == 0)
            return wake_fd >= 0 && openai_http_stop ? 0 : 400;
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
    return authorization_size == prefix_size + key_size &&
           memcmp(request->authorization, prefix, prefix_size) == 0 &&
           memcmp(request->authorization + prefix_size,
                  config->api_key, key_size) == 0;
}

static int openai_http_read_body(
    int fd, int wake_fd,
    const char *head, size_t data_size, size_t header_size,
    size_t body_size, char **body) {
    *body = malloc(body_size + 1);
    if (!*body) return -1;
    size_t buffered = data_size - header_size;
    if (buffered > body_size) buffered = body_size;
    if (buffered) memcpy(*body, head + header_size, buffered);
    size_t offset = buffered;
    while (offset < body_size) {
        if (!openai_http_wait_readable(fd, wake_fd)) {
            free(*body);
            *body = NULL;
            return -2;
        }
        ssize_t received = recv(fd, *body + offset, body_size - offset, 0);
        if (received > 0) {
            offset += (size_t)received;
            continue;
        }
        if (received < 0 && errno == EINTR) {
            if (wake_fd >= 0 && openai_http_stop) {
                free(*body);
                *body = NULL;
                return -2;
            }
            continue;
        }
        if (received == 0) {
            free(*body);
            *body = NULL;
            return wake_fd >= 0 && openai_http_stop ? -2 : 0;
        }
        free(*body);
        *body = NULL;
        return -2;
    }
    (*body)[body_size] = 0;
    return 1;
}

static int openai_content_sink(
    int token, const char *piece, size_t piece_size, void *user_data) {
    (void)token;
    OpenAIContentBuffer *buffer = user_data;
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

static FILE *openai_sse_event_stream(char **event, size_t *event_size) {
    *event = NULL;
    *event_size = 0;
    return open_memstream(event, event_size);
}

static int openai_sse_send_event(
    int fd, char *event, size_t event_size, int formatted) {
    int ok = formatted &&
             openai_http_send_all(fd, "data: ", 6) &&
             openai_http_send_all(fd, event, event_size) &&
             openai_http_send_all(fd, "\n\n", 2);
    free(event);
    return ok;
}

static int openai_sse_send_role(int fd, const char *model) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    int formatted = fputs("{\"id\":\"" OPENAI_HTTP_ID
                          "\",\"object\":\"chat.completion.chunk\","
                          "\"created\":0,\"model\":", output) != EOF &&
                    openai_json_string(output, model) &&
                    fputs(",\"choices\":[{\"index\":0,\"delta\":{"
                          "\"role\":\"assistant\"},"
                          "\"finish_reason\":null}]}", output) != EOF;
    if (fclose(output) != 0) formatted = 0;
    return openai_sse_send_event(fd, event, event_size, formatted);
}

static int openai_sse_send_content(
    int fd, const char *model, const char *piece, size_t piece_size) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    int formatted = fputs("{\"id\":\"" OPENAI_HTTP_ID
                          "\",\"object\":\"chat.completion.chunk\","
                          "\"created\":0,\"model\":", output) != EOF &&
                    openai_json_string(output, model) &&
                    fputs(",\"choices\":[{\"index\":0,\"delta\":{"
                          "\"content\":", output) != EOF &&
                    openai_json_string_n(output, piece, piece_size) &&
                    fputs("},\"finish_reason\":null}]}", output) != EOF;
    if (fclose(output) != 0) formatted = 0;
    return openai_sse_send_event(fd, event, event_size, formatted);
}

static int openai_sse_token_sink(
    int token, const char *piece, size_t piece_size, void *user_data) {
    (void)token;
    OpenAISseSink *sink = user_data;
    if (sink->failed || (!piece && piece_size)) return 0;
    if (!openai_sse_send_content(sink->fd, sink->model, piece, piece_size)) {
        sink->failed = 1;
        return 0;
    }
    return 1;
}

static int openai_sse_send_terminal(
    int fd, const char *model, const OpenAIGenerationResult *result) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    const char *reason = result->finish_reason ? result->finish_reason : "stop";
    int formatted = fputs("{\"id\":\"" OPENAI_HTTP_ID
                          "\",\"object\":\"chat.completion.chunk\","
                          "\"created\":0,\"model\":", output) != EOF &&
                    openai_json_string(output, model) &&
                    fputs(",\"choices\":[{\"index\":0,\"delta\":{},"
                          "\"finish_reason\":", output) != EOF &&
                    openai_json_string(output, reason) &&
                    fputs("}]}", output) != EOF;
    if (fclose(output) != 0) formatted = 0;
    return openai_sse_send_event(fd, event, event_size, formatted);
}

static int openai_sse_send_usage(
    int fd, const char *model, const OpenAIGenerationResult *result) {
    char *event;
    size_t event_size;
    FILE *output = openai_sse_event_stream(&event, &event_size);
    if (!output) return 0;
    long long total_tokens =
        (long long)result->prompt_tokens + result->completion_tokens;
    int formatted = fputs("{\"id\":\"" OPENAI_HTTP_ID
                          "\",\"object\":\"chat.completion.chunk\","
                          "\"created\":0,\"model\":", output) != EOF &&
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
    return openai_sse_send_event(fd, event, event_size, formatted);
}

static int openai_http_send_sse_headers(int fd) {
    static const char headers[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n\r\n";
    return openai_http_send_all(fd, headers, sizeof(headers) - 1);
}

static int openai_http_run_completion(
    int fd, const OpenAIHttpConfig *config, OpenAIGenerateHandler handler,
    void *user_data, char *body) {
    OpenAIChatRequest request = {0};
    char request_error[256] = {0};
    if (!openai_chat_request_parse(
            body, config->model_name, &request,
            request_error, sizeof(request_error))) {
        return openai_http_send_error(
            fd, 400, request_error[0] ? request_error : "invalid request",
            "invalid_request", NULL);
    }
    if (!handler) {
        openai_chat_request_free(&request);
        return openai_http_send_error(
            fd, 500, "generation handler is unavailable", "server_error", NULL);
    }

    OpenAIGenerationResult result = {0};
    char generation_error[256] = {0};
    if (request.stream) {
        int ok = openai_http_send_sse_headers(fd) &&
                 openai_sse_send_role(fd, request.model);
        OpenAISseSink sink = {.fd = fd, .model = request.model, .failed = !ok};
        int generated = ok && handler(
            user_data, &request, openai_sse_token_sink, &sink,
            &result, generation_error, sizeof(generation_error));
        if (generated && !sink.failed)
            ok = openai_sse_send_terminal(fd, request.model, &result);
        else
            ok = 0;
        if (ok && request.include_usage)
            ok = openai_sse_send_usage(fd, request.model, &result);
        if (ok) ok = openai_http_send_all(fd, "data: [DONE]\n\n", 14);
        openai_chat_request_free(&request);
        return ok;
    }

    OpenAIContentBuffer content = {0};
    int generated = handler(
        user_data, &request, openai_content_sink, &content,
        &result, generation_error, sizeof(generation_error));
    if (!generated || content.failed) {
        free(content.data);
        openai_chat_request_free(&request);
        return openai_http_send_error(
            fd, 500,
            generation_error[0] ? generation_error : "generation failed",
            "generation_failed", NULL);
    }
    if (!content.data) {
        content.data = openai_strdup("");
        if (!content.data) {
            openai_chat_request_free(&request);
            return openai_http_send_error(
                fd, 500, "out of memory formatting response",
                "server_error", NULL);
        }
    }
    char *json = NULL;
    size_t json_size = 0;
    int formatted = openai_format_completion_json(
        OPENAI_HTTP_ID, request.model, content.data,
        &result, &json, &json_size);
    free(content.data);
    openai_chat_request_free(&request);
    if (!formatted)
        return openai_http_send_error(
            fd, 500, "could not format response", "server_error", NULL);
    int ok = openai_http_send_response(
        fd, 200, "application/json", NULL, json, json_size);
    free(json);
    return ok;
}

static int openai_http_handle_connection_internal(
    int fd, const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data, int wake_fd) {
    if (fd < 0 || !config || !config->model_name || !config->model_name[0])
        return 0;
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif

    char head[OPENAI_HTTP_HEADER_LIMIT + 1];
    size_t data_size = 0;
    size_t header_size = 0;
    int read_status = openai_http_read_head(
        fd, wake_fd, head, &data_size, &header_size);
    if (read_status == 431)
        return openai_http_send_error(
            fd, 431, "request headers exceed 32 KiB",
            "headers_too_large", NULL);
    if (read_status == 400)
        return openai_http_send_error(
            fd, 400, "incomplete HTTP headers", "invalid_http", NULL);
    if (read_status != 1) return 0;

    OpenAIHttpRequest request = {0};
    int parsed = openai_http_parse_head(head, header_size, &request);
    if (parsed < 0)
        return openai_http_send_error(
            fd, 500, "out of memory parsing headers", "server_error", NULL);
    if (!parsed)
        return openai_http_send_error(
            fd, 400, "malformed HTTP request", "invalid_http", NULL);

    int ok = 0;
    if (request.has_transfer_encoding) {
        ok = openai_http_send_error(
            fd, 400, "Transfer-Encoding is not supported",
            "unsupported_transfer_encoding", NULL);
        goto done;
    }
    if (request.has_content_length &&
        request.content_length > OPENAI_HTTP_BODY_LIMIT) {
        ok = openai_http_send_error(
            fd, 413, "request body exceeds 8 MiB", "body_too_large", NULL);
        goto done;
    }
    if (strncmp(request.path, "/v1", 3) == 0 &&
        (request.path[3] == 0 || request.path[3] == '/') &&
        !openai_http_authorized(config, &request)) {
        ok = openai_http_send_error(
            fd, 401, "invalid API key", "invalid_api_key",
            "WWW-Authenticate: Bearer\r\n");
        goto done;
    }

    if (strcmp(request.path, "/v1/models") == 0) {
        if (strcmp(request.method, "GET") != 0) {
            ok = openai_http_send_error(
                fd, 405, "method not allowed", "method_not_allowed",
                "Allow: GET\r\n");
            goto done;
        }
        char *json = NULL;
        size_t json_size = 0;
        if (!openai_format_models_json(config->model_name, &json, &json_size)) {
            ok = openai_http_send_error(
                fd, 500, "could not format response", "server_error", NULL);
            goto done;
        }
        ok = openai_http_send_response(
            fd, 200, "application/json", NULL, json, json_size);
        free(json);
        goto done;
    }

    if (strcmp(request.path, "/v1/chat/completions") == 0) {
        if (strcmp(request.method, "POST") != 0) {
            ok = openai_http_send_error(
                fd, 405, "method not allowed", "method_not_allowed",
                "Allow: POST\r\n");
            goto done;
        }
        if (!request.has_content_length) {
            ok = openai_http_send_error(
                fd, 400, "Content-Length is required",
                "missing_content_length", NULL);
            goto done;
        }
        char *body = NULL;
        int body_status = openai_http_read_body(
            fd, wake_fd, head, data_size, header_size,
            request.content_length, &body);
        if (body_status == 0) {
            ok = openai_http_send_error(
                fd, 400, "request body is shorter than Content-Length",
                "incomplete_body", NULL);
            goto done;
        }
        if (body_status < 0) {
            if (body_status == -1)
                ok = openai_http_send_error(
                    fd, 500, "out of memory reading request body",
                    "server_error", NULL);
            goto done;
        }
        if (memchr(body, 0, request.content_length)) {
            free(body);
            ok = openai_http_send_error(
                fd, 400, "request body contains a NUL byte",
                "invalid_json", NULL);
            goto done;
        }
        ok = openai_http_run_completion(
            fd, config, handler, user_data, body);
        free(body);
        goto done;
    }

    ok = openai_http_send_error(
        fd, 404, "route not found", "not_found", NULL);

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
    int fd = (int)openai_http_wakeup_fd;
    if (fd >= 0) {
        const unsigned char byte = 1;
        (void)write(fd, &byte, 1);
    }
    errno = saved_errno;
}

static int openai_http_configure_pipe(int pipe_fds[2]) {
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
            pipe_fds[0] = -1;
            pipe_fds[1] = -1;
            return 0;
        }
    }
    return 1;
}

int openai_http_serve(
    const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!config || !config->host || !config->host[0] ||
        config->port < 1 || config->port > 65535 ||
        !config->model_name || !config->model_name[0] || !handler) {
        openai_error(error, error_size, "invalid HTTP server configuration");
        return 0;
    }

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

    int wakeup[2] = {-1, -1};
    if (!openai_http_configure_pipe(wakeup)) {
        close(listener);
        openai_error(error, error_size, "could not create HTTP wakeup pipe");
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
    openai_http_stop = 0;
    openai_http_wakeup_fd = wakeup[1];
    if (sigaction(SIGINT, &action, &old_int) != 0) {
        openai_http_wakeup_fd = -1;
        close(wakeup[0]);
        close(wakeup[1]);
        close(listener);
        openai_error(error, error_size, "could not install signal handlers");
        return 0;
    }
    if (sigaction(SIGTERM, &action, &old_term) != 0) {
        openai_http_wakeup_fd = -1;
        sigaction(SIGINT, &old_int, NULL);
        openai_http_drain_wakeup(wakeup[0]);
        close(wakeup[0]);
        close(wakeup[1]);
        close(listener);
        openai_error(error, error_size, "could not install signal handlers");
        return 0;
    }
    int ok = 1;
    struct pollfd fds[2] = {
        {.fd = listener, .events = POLLIN},
        {.fd = wakeup[0], .events = POLLIN},
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
            openai_http_drain_wakeup(wakeup[0]);
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
            connection, config, handler, user_data, wakeup[0]);
        close(connection);
    }

    openai_http_wakeup_fd = -1;
    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGTERM, &old_term, NULL);
    openai_http_drain_wakeup(wakeup[0]);
    close(wakeup[0]);
    close(wakeup[1]);
    close(listener);
    return ok;
}
