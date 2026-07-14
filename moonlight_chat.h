#ifndef MOONLIGHT_CHAT_H
#define MOONLIGHT_CHAT_H

typedef struct {
    const char *model_dir;
    const char *prompt;
    const char *system_prompt;
    int max_context;
    int max_new_tokens;
    float temperature;
    float top_p;
} MoonlightChatOptions;

typedef struct {
    const char *model_dir;
    int max_context;
    int max_new_tokens;
    const char *host;
    int port;
    const char *api_key;
    const char *served_model_name;
} MoonlightServeOptions;

int moonlight_metal_model_dir(const char *model_dir);
int moonlight_chat_run(const MoonlightChatOptions *options);
int moonlight_serve_run(const MoonlightServeOptions *options);

#endif
