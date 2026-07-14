#ifndef DEEPSEEK_V4_SERVE_H
#define DEEPSEEK_V4_SERVE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    char *role;
    char *content;
} DeepSeekV4ServeMessage;

typedef struct {
    char *id_json;
    char *prompt;
    char *system;
    DeepSeekV4ServeMessage *messages;
    size_t message_count;
    int max_tokens;
    float temperature;
    float top_p;
    int draft;
} DeepSeekV4ServeRequest;

typedef struct {
    const char *id_json;
    const char *text;
    int prompt_tokens;
    int cached_tokens;
    int completion_tokens;
    double prompt_ms;
    double decode_ms;
    int cache_hit;
    int cache_prefix_tokens;
    uint64_t cache_entries;
    uint64_t cache_bytes;
    const char *error_code;
    const char *error_message;
} DeepSeekV4ServeResponse;

typedef int (*DeepSeekV4ServeHandler)(
    void *user_data, const DeepSeekV4ServeRequest *request,
    DeepSeekV4ServeResponse *response, char *error, size_t error_size);

int deepseek_v4_serve_parse_request(
    const char *line, DeepSeekV4ServeRequest *request,
    char *error, size_t error_size);
void deepseek_v4_serve_request_free(DeepSeekV4ServeRequest *request);
int deepseek_v4_serve_write_response(
    FILE *output, const DeepSeekV4ServeResponse *response);
int deepseek_v4_serve_stdio(
    FILE *input, FILE *output, DeepSeekV4ServeHandler handler, void *user_data);
int deepseek_v4_openai_effective_draft(
    int default_draft, float temperature);
int deepseek_v4_openai_is_client_error(const char *error);

#endif
