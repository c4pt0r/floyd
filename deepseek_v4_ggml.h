#ifndef DEEPSEEK_V4_GGML_H
#define DEEPSEEK_V4_GGML_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeepSeekV4GgmlSession DeepSeekV4GgmlSession;

typedef int (*DeepSeekV4GgmlTokenCallback)(int token, const char *piece,
                                           size_t piece_size, void *user_data);

typedef struct {
    double load_ms;
    double prompt_ms;
    double decode_ms;
    int prompt_tokens;
    int generated_tokens;
} DeepSeekV4GgmlStats;

const char *deepseek_v4_ggml_backend_name(void);
int deepseek_v4_ggml_find_model(const char *checkpoint_dir,
                                char *model_path, size_t model_path_size,
                                char *error, size_t error_size);
DeepSeekV4GgmlSession *deepseek_v4_ggml_open(
    const char *checkpoint_dir, int max_context,
    char *error, size_t error_size);
int deepseek_v4_ggml_reset(DeepSeekV4GgmlSession *session);
int deepseek_v4_ggml_generate_user(
    DeepSeekV4GgmlSession *session, const char *text, int first_turn,
    int max_new_tokens, DeepSeekV4GgmlTokenCallback callback, void *user_data,
    DeepSeekV4GgmlStats *stats, char *error, size_t error_size);
void deepseek_v4_ggml_close(DeepSeekV4GgmlSession *session);

#ifdef __cplusplus
}
#endif

#endif
