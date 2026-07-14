#ifndef OPENAI_HTTP_H
#define OPENAI_HTTP_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *role;
    char *content;
} OpenAIMessage;

typedef struct {
    char *model;
    OpenAIMessage *messages;
    size_t message_count;
    int max_tokens;
    float temperature;
    float top_p;
    int stream;
    int include_usage;
} OpenAIChatRequest;

typedef struct {
    int prompt_tokens;
    int cached_tokens;
    int completion_tokens;
    double prompt_ms;
    double decode_ms;
    uint64_t cache_entries;
    uint64_t cache_bytes;
    const char *finish_reason;
} OpenAIGenerationResult;

typedef enum {
    OPENAI_GENERATE_INTERNAL_ERROR = 0,
    OPENAI_GENERATE_OK = 1,
    OPENAI_GENERATE_CLIENT_ERROR = 2,
} OpenAIGenerateStatus;

typedef int (*OpenAITokenSink)(int token, const char *piece,
                               size_t piece_size, void *user_data);
typedef int (*OpenAIGenerateHandler)(
    void *user_data, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size);

typedef struct {
    const char *host;
    int port;
    const char *api_key;
    const char *model_name;
    int io_timeout_ms;
} OpenAIHttpConfig;

int openai_chat_request_parse(
    const char *body, const char *served_model, OpenAIChatRequest *request,
    char *error, size_t error_size);
void openai_chat_request_free(OpenAIChatRequest *request);

int openai_format_models_json(
    const char *model, char **json, size_t *json_size);
int openai_format_completion_json(
    const char *id, const char *model, const char *content, size_t content_size,
    const OpenAIGenerationResult *result, char **json, size_t *json_size);
int openai_format_error_json(
    const char *message, const char *type, const char *param, const char *code,
    char **json, size_t *json_size);

int openai_http_handle_connection(
    int fd, const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data);
int openai_http_serve(
    const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data,
    char *error, size_t error_size);

#endif
