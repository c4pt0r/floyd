#ifndef DEEPSEEK_V4_CHAT_H
#define DEEPSEEK_V4_CHAT_H

typedef struct {
    const char *model_dir;
    int max_context;
    int max_new_tokens;
    int use_spec;
} DeepSeekV4ChatOptions;

int deepseek_v4_model_dir(const char *model_dir);
int deepseek_v4_chat_run(const DeepSeekV4ChatOptions *options);

#endif
