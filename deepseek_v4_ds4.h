#ifndef DEEPSEEK_V4_DS4_H
#define DEEPSEEK_V4_DS4_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeepSeekV4Ds4Session DeepSeekV4Ds4Session;

typedef struct {
    int draft_tokens;
    float margin;
} DeepSeekV4Ds4SpecConfig;

typedef int (*DeepSeekV4Ds4TokenCallback)(int token, const char *piece,
                                          size_t piece_size, void *user_data);

typedef struct {
    double load_ms;
    double prompt_ms;
    double decode_ms;
    int prompt_tokens;
    int generated_tokens;
    int speculative_rounds;
    int speculative_tokens;
} DeepSeekV4Ds4Stats;

const char *deepseek_v4_ds4_backend_name(void);
int deepseek_v4_ds4_spec_config_from_env(
    DeepSeekV4Ds4SpecConfig *config, char *error, size_t error_size);
int deepseek_v4_ds4_find_model(const char *checkpoint_dir,
                               char *model_path, size_t model_path_size,
                               char *error, size_t error_size);
DeepSeekV4Ds4Session *deepseek_v4_ds4_open(
    const char *model_path, int max_context, int use_spec,
    char *error, size_t error_size);
int deepseek_v4_ds4_reset(DeepSeekV4Ds4Session *session);
int deepseek_v4_ds4_generate_user(
    DeepSeekV4Ds4Session *session, const char *text, int max_new_tokens,
    DeepSeekV4Ds4TokenCallback callback, void *user_data,
    DeepSeekV4Ds4Stats *stats, char *error, size_t error_size);
void deepseek_v4_ds4_close(DeepSeekV4Ds4Session *session);

#ifdef __cplusplus
}
#endif

#endif
