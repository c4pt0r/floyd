#ifndef DEEPSEEK_V4_GGML_H
#define DEEPSEEK_V4_GGML_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *deepseek_v4_ggml_backend_name(void);
int deepseek_v4_ggml_find_model(const char *checkpoint_dir,
                                char *model_path, size_t model_path_size,
                                char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
