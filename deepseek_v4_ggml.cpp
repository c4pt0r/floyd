#include "deepseek_v4_ggml.h"

#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int deepseek_v4_ggml_regular_file(const char *path) {
    struct stat info;
    return path && stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static int deepseek_v4_ggml_copy(char *destination, size_t capacity,
                                 const char *source) {
    if (!destination || !capacity || !source) return 0;
    int count = snprintf(destination, capacity, "%s", source);
    return count >= 0 && (size_t)count < capacity;
}

static void deepseek_v4_ggml_error(char *error, size_t capacity,
                                   const char *format, const char *value) {
    if (error && capacity) snprintf(error, capacity, format, value ? value : "");
}

const char *deepseek_v4_ggml_backend_name(void) {
    return "metal-ggml";
}

int deepseek_v4_ggml_find_model(const char *checkpoint_dir,
                                char *model_path, size_t model_path_size,
                                char *error, size_t error_size) {
    if (!checkpoint_dir || !model_path || !model_path_size) {
        deepseek_v4_ggml_error(error, error_size,
                               "invalid DeepSeek V4 checkpoint: %s", checkpoint_dir);
        return 0;
    }

    const char *override_path = getenv("FLOYD_DEEPSEEK_V4_GGUF");
    if (override_path && *override_path) {
        if (!deepseek_v4_ggml_regular_file(override_path)) {
            deepseek_v4_ggml_error(error, error_size,
                                   "FLOYD_DEEPSEEK_V4_GGUF is not a file: %s",
                                   override_path);
            return 0;
        }
        if (!deepseek_v4_ggml_copy(model_path, model_path_size, override_path)) {
            deepseek_v4_ggml_error(error, error_size,
                                   "FLOYD_DEEPSEEK_V4_GGUF path is too long: %s",
                                   override_path);
            return 0;
        }
        if (error && error_size) error[0] = 0;
        return 1;
    }

    char pattern[4096];
    const char *patterns[] = {
        "%s-GGUF/model-*-00001-of-*.gguf",
        "%s-GGUF/model-*.gguf",
    };
    for (size_t index = 0; index < sizeof(patterns) / sizeof(patterns[0]); index++) {
        int count = snprintf(pattern, sizeof(pattern), patterns[index], checkpoint_dir);
        if (count < 0 || (size_t)count >= sizeof(pattern)) break;
        glob_t matches = {};
        int result = glob(pattern, 0, NULL, &matches);
        if (result == 0 && matches.gl_pathc > 0 &&
            deepseek_v4_ggml_regular_file(matches.gl_pathv[0])) {
            int copied = deepseek_v4_ggml_copy(model_path, model_path_size,
                                               matches.gl_pathv[0]);
            globfree(&matches);
            if (!copied) {
                deepseek_v4_ggml_error(error, error_size,
                                       "prepared GGUF path is too long: %s", checkpoint_dir);
                return 0;
            }
            if (error && error_size) error[0] = 0;
            return 1;
        }
        globfree(&matches);
    }

    deepseek_v4_ggml_error(
        error, error_size,
        "prepared GGUF not found; run make prepare-deepseek-v4-gguf DSPARK=%s",
        checkpoint_dir);
    return 0;
}
