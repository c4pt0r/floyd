#define _GNU_SOURCE
#include "deepseek_v4_serve.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static void serve_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static char *serve_strdup(const char *value) {
    if (!value) return NULL;
    size_t size = strlen(value) + 1;
    char *copy = malloc(size);
    if (copy) memcpy(copy, value, size);
    return copy;
}

static int serve_json_string(FILE *output, const char *value) {
    if (fputc('"', output) == EOF) return 0;
    for (const unsigned char *p = (const unsigned char *)(value ? value : "");
         *p; p++) {
        switch (*p) {
            case '"': if (fputs("\\\"", output) == EOF) return 0; break;
            case '\\': if (fputs("\\\\", output) == EOF) return 0; break;
            case '\b': if (fputs("\\b", output) == EOF) return 0; break;
            case '\f': if (fputs("\\f", output) == EOF) return 0; break;
            case '\n': if (fputs("\\n", output) == EOF) return 0; break;
            case '\r': if (fputs("\\r", output) == EOF) return 0; break;
            case '\t': if (fputs("\\t", output) == EOF) return 0; break;
            default:
                if (*p < 0x20) {
                    if (fprintf(output, "\\u%04x", (unsigned)*p) < 0) return 0;
                } else if (fputc(*p, output) == EOF) return 0;
                break;
        }
    }
    return fputc('"', output) != EOF;
}

static char *serve_id_json(const jval *id) {
    if (!id) return NULL;
    if (id->t == J_STR) {
        char *encoded = NULL;
        size_t size = 0;
        FILE *stream = open_memstream(&encoded, &size);
        if (!stream) return NULL;
        int ok = serve_json_string(stream, id->str);
        if (fclose(stream) != 0) ok = 0;
        if (!ok) {
            free(encoded);
            return NULL;
        }
        return encoded;
    }
    if (id->t == J_NUM && isfinite(id->num) && floor(id->num) == id->num &&
        id->num >= -9007199254740991.0 && id->num <= 9007199254740991.0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.0f", id->num);
        return serve_strdup(buf);
    }
    return NULL;
}

static int serve_int_field(jval *root, const char *name, int missing,
                           int min_value, int max_value, int *out,
                           char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value) { *out = missing; return 1; }
    if (value->t != J_NUM || !isfinite(value->num) || floor(value->num) != value->num ||
        value->num < min_value || value->num > max_value) {
        char message[128];
        snprintf(message, sizeof(message), "%s must be an integer in %d..%d",
                 name, min_value, max_value);
        serve_error(error, error_size, message);
        return 0;
    }
    *out = (int)value->num;
    return 1;
}

static int serve_float_field(jval *root, const char *name, float missing,
                             double min_value, double max_value, int min_open,
                             float *out, char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value) { *out = missing; return 1; }
    if (value->t != J_NUM || !isfinite(value->num) ||
        (min_open ? value->num <= min_value : value->num < min_value) ||
        value->num > max_value) {
        char message[128];
        snprintf(message, sizeof(message), "%s is outside its supported range", name);
        serve_error(error, error_size, message);
        return 0;
    }
    *out = (float)value->num;
    return 1;
}

void deepseek_v4_serve_request_free(DeepSeekV4ServeRequest *request) {
    if (!request) return;
    free(request->id_json);
    free(request->prompt);
    free(request->system);
    for (size_t i = 0; i < request->message_count; i++) {
        free(request->messages[i].role);
        free(request->messages[i].content);
    }
    free(request->messages);
    memset(request, 0, sizeof(*request));
}

int deepseek_v4_serve_parse_request(
    const char *line, DeepSeekV4ServeRequest *request,
    char *error, size_t error_size) {
    if (!request) return 0;
    memset(request, 0, sizeof(*request));
    request->max_tokens = -1;
    request->temperature = 0.0f;
    request->top_p = 1.0f;
    request->draft = -1;

    char json_error[256] = {0};
    jval *root = json_parse_full(line, json_error, sizeof(json_error));
    if (!root) {
        serve_error(error, error_size, json_error);
        return 0;
    }
    if (root->t != J_OBJ) {
        serve_error(error, error_size, "request must be a JSON object");
        json_free(root);
        return 0;
    }

    request->id_json = serve_id_json(json_get(root, "id"));
    if (!request->id_json) {
        serve_error(error, error_size, "id must be a string or integer");
        goto fail;
    }
    jval *prompt = json_get(root, "prompt");
    jval *messages = json_get(root, "messages");
    if (!!prompt == !!messages) {
        serve_error(error, error_size, "request requires exactly one of prompt or messages");
        goto fail;
    }
    jval *system = json_get(root, "system");
    if (system && system->t != J_STR) {
        serve_error(error, error_size, "system must be a string");
        goto fail;
    }
    if (prompt) {
        if (prompt->t != J_STR) {
            serve_error(error, error_size, "prompt must be a string");
            goto fail;
        }
        request->prompt = serve_strdup(prompt->str);
        request->system = system ? serve_strdup(system->str) : NULL;
        if (!request->prompt || (system && !request->system)) goto oom;
    } else {
        if (system) {
            serve_error(error, error_size, "system cannot be combined with messages");
            goto fail;
        }
        if (messages->t != J_ARR || messages->len <= 0) {
            serve_error(error, error_size, "messages must be a non-empty array");
            goto fail;
        }
        request->messages = calloc((size_t)messages->len, sizeof(*request->messages));
        if (!request->messages) goto oom;
        request->message_count = (size_t)messages->len;
        for (int i = 0; i < messages->len; i++) {
            jval *item = messages->kids[i];
            jval *role = json_get(item, "role");
            jval *content = json_get(item, "content");
            if (!item || item->t != J_OBJ || !role || role->t != J_STR ||
                !content || content->t != J_STR ||
                (strcmp(role->str, "system") && strcmp(role->str, "user") &&
                 strcmp(role->str, "assistant"))) {
                serve_error(error, error_size,
                            "each message requires a supported role and string content");
                goto fail;
            }
            request->messages[i].role = serve_strdup(role->str);
            request->messages[i].content = serve_strdup(content->str);
            if (!request->messages[i].role || !request->messages[i].content) goto oom;
        }
    }
    if (!serve_int_field(root, "max_tokens", -1, 1, INT_MAX,
                         &request->max_tokens, error, error_size) ||
        !serve_float_field(root, "temperature", 0.0f, 0.0, 2.0, 0,
                           &request->temperature, error, error_size) ||
        !serve_float_field(root, "top_p", 1.0f, 0.0, 1.0, 1,
                           &request->top_p, error, error_size) ||
        !serve_int_field(root, "draft", -1, 0, 16,
                         &request->draft, error, error_size)) goto fail;

    json_free(root);
    return 1;

oom:
    serve_error(error, error_size, "out of memory parsing request");
fail:
    json_free(root);
    deepseek_v4_serve_request_free(request);
    return 0;
}

int deepseek_v4_serve_write_response(
    FILE *output, const DeepSeekV4ServeResponse *response) {
    if (!output || !response) return 0;
    int uncached = response->prompt_tokens - response->cached_tokens;
    if (uncached < 0) uncached = 0;
    if (fprintf(output, "{\"id\":%s,\"text\":",
                response->id_json ? response->id_json : "null") < 0 ||
        !serve_json_string(output, response->text ? response->text : "") ||
        fprintf(output,
                ",\"usage\":{\"prompt_tokens\":%d,\"cached_tokens\":%d,"
                "\"uncached_tokens\":%d,\"completion_tokens\":%d},"
                "\"stats\":{\"prompt_ms\":%.3f,\"decode_ms\":%.3f,"
                "\"cache_entries\":%llu,\"cache_bytes\":%llu},"
                "\"cache_hit\":%s,\"cache_prefix_tokens\":%d,\"error\":",
                response->prompt_tokens, response->cached_tokens, uncached,
                response->completion_tokens, response->prompt_ms,
                response->decode_ms,
                (unsigned long long)response->cache_entries,
                (unsigned long long)response->cache_bytes,
                response->cache_hit ? "true" : "false",
                response->cache_prefix_tokens) < 0) return 0;
    if (response->error_code) {
        if (fputs("{\"code\":", output) == EOF ||
            !serve_json_string(output, response->error_code) ||
            fputs(",\"message\":", output) == EOF ||
            !serve_json_string(output, response->error_message ? response->error_message : "") ||
            fputc('}', output) == EOF) return 0;
    } else if (fputs("null", output) == EOF) return 0;
    return fputs("}\n", output) != EOF && fflush(output) == 0;
}

int deepseek_v4_serve_stdio(
    FILE *input, FILE *output, DeepSeekV4ServeHandler handler, void *user_data) {
    if (!input || !output || !handler) return 1;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t size;
    int status = 0;
    while ((size = getline(&line, &capacity, input)) >= 0) {
        while (size > 0 && (line[size - 1] == '\n' || line[size - 1] == '\r'))
            line[--size] = 0;
        if (size == 0) continue;
        DeepSeekV4ServeRequest request = {0};
        DeepSeekV4ServeResponse response = {0};
        char error[512] = {0};
        if (!deepseek_v4_serve_parse_request(
                line, &request, error, sizeof(error))) {
            response.error_code = "invalid_request";
            response.error_message = error;
        } else {
            response.id_json = request.id_json;
            if (!handler(user_data, &request, &response, error, sizeof(error))) {
                if (!response.error_code) response.error_code = "inference_error";
                if (!response.error_message)
                    response.error_message = error[0] ? error : "request failed";
            }
        }
        if (!deepseek_v4_serve_write_response(output, &response)) {
            status = 1;
            deepseek_v4_serve_request_free(&request);
            break;
        }
        deepseek_v4_serve_request_free(&request);
    }
    if (ferror(input)) status = 1;
    free(line);
    return status;
}
