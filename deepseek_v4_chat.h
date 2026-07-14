#ifndef DEEPSEEK_V4_CHAT_H
#define DEEPSEEK_V4_CHAT_H

#include <stdint.h>

typedef struct {
    const char *model_dir;
    const char *prompt;
    int max_context;
    int max_new_tokens;
    int use_spec;
} DeepSeekV4ChatOptions;

typedef struct {
    const char *model_dir;
    int max_context;
    int max_new_tokens;
    int draft;
    uint64_t prefix_cache_bytes;
    int stdio;
    const char *host;
    int port;
    const char *api_key;
    const char *served_model_name;
} DeepSeekV4ServeOptions;

int deepseek_v4_model_dir(const char *model_dir);
int deepseek_v4_chat_run(const DeepSeekV4ChatOptions *options);
int deepseek_v4_serve_run(const DeepSeekV4ServeOptions *options);

#endif
