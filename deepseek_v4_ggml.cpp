#include "deepseek_v4_ggml.h"

#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef FLOYD_DEEPSEEK_V4_GGML
#include <algorithm>
#include <new>
#include <string>
#include <vector>

#include "deepseek_v4_chat_format.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#endif

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

#ifdef FLOYD_DEEPSEEK_V4_GGML

struct DeepSeekV4GgmlSession {
    llama_model *model;
    llama_context *context;
    const llama_vocab *vocab;
    double load_ms;
};

static int deepseek_v4_ggml_initialized;

static void deepseek_v4_ggml_log(enum ggml_log_level level,
                                 const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static int deepseek_v4_ggml_initialize_backend(char *error, size_t error_size) {
    if (!deepseek_v4_ggml_initialized) {
        llama_backend_init();
        ggml_backend_load_all();
        llama_log_set(deepseek_v4_ggml_log, NULL);
        deepseek_v4_ggml_initialized = 1;
    }
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const char *name = ggml_backend_dev_name(device);
        const char *description = ggml_backend_dev_description(device);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU ||
             type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            ((name && strstr(name, "Metal")) ||
             (description && (strstr(description, "Metal") ||
                              strstr(description, "Apple")))))
            return 1;
    }
    deepseek_v4_ggml_error(error, error_size,
                           "Metal ggml backend is unavailable: %s",
                           "rebuild llama.cpp with GGML_METAL=ON");
    return 0;
}

DeepSeekV4GgmlSession *deepseek_v4_ggml_open(
    const char *checkpoint_dir, int max_context,
    char *error, size_t error_size) {
    if (max_context <= 0) {
        deepseek_v4_ggml_error(error, error_size,
                               "invalid DeepSeek V4 context size: %s", "must be positive");
        return NULL;
    }
    char model_path[4096];
    if (!deepseek_v4_ggml_find_model(checkpoint_dir, model_path,
                                     sizeof(model_path), error, error_size) ||
        !deepseek_v4_ggml_initialize_backend(error, error_size))
        return NULL;

    DeepSeekV4GgmlSession *session = new (std::nothrow) DeepSeekV4GgmlSession{};
    if (!session) {
        deepseek_v4_ggml_error(error, error_size,
                               "failed to allocate DeepSeek V4 session: %s", "out of memory");
        return NULL;
    }

    int64_t started = ggml_time_us();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.use_mmap = true;
    session->model = llama_model_load_from_file(model_path, model_params);
    if (!session->model) {
        deepseek_v4_ggml_error(error, error_size,
                               "failed to load prepared DeepSeek V4 GGUF: %s", model_path);
        delete session;
        return NULL;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = (uint32_t)max_context;
    context_params.n_batch = (uint32_t)std::min(max_context, 2048);
    context_params.n_ubatch = context_params.n_batch;
    context_params.n_seq_max = 1;
    context_params.offload_kqv = true;
    context_params.op_offload = true;
    context_params.no_perf = false;
    session->context = llama_init_from_model(session->model, context_params);
    if (!session->context) {
        deepseek_v4_ggml_error(error, error_size,
                               "failed to initialize DeepSeek V4 ggml context: %s", model_path);
        llama_model_free(session->model);
        delete session;
        return NULL;
    }
    session->vocab = llama_model_get_vocab(session->model);
    session->load_ms = (ggml_time_us() - started) / 1000.0;
    if (error && error_size) error[0] = 0;
    return session;
}

int deepseek_v4_ggml_reset(DeepSeekV4GgmlSession *session) {
    if (!session || !session->context) return 0;
    llama_memory_clear(llama_get_memory(session->context), true);
    llama_perf_context_reset(session->context);
    return 1;
}

static int deepseek_v4_ggml_tokenize(DeepSeekV4GgmlSession *session,
                                     const char *text,
                                     std::vector<llama_token> *tokens) {
    int count = llama_tokenize(session->vocab, text, strlen(text),
                               NULL, 0, false, true);
    if (count >= 0) return 0;
    tokens->resize((size_t)-count);
    count = llama_tokenize(session->vocab, text, strlen(text),
                           tokens->data(), (int32_t)tokens->size(), false, true);
    if (count <= 0) return 0;
    tokens->resize((size_t)count);
    return 1;
}

static int deepseek_v4_ggml_piece(DeepSeekV4GgmlSession *session,
                                  llama_token token, std::string *piece) {
    std::vector<char> buffer(256);
    int count = llama_token_to_piece(session->vocab, token, buffer.data(),
                                     (int32_t)buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize((size_t)-count);
        count = llama_token_to_piece(session->vocab, token, buffer.data(),
                                     (int32_t)buffer.size(), 0, true);
    }
    if (count < 0) return 0;
    piece->assign(buffer.data(), (size_t)count);
    return 1;
}

static int deepseek_v4_ggml_decode(DeepSeekV4GgmlSession *session,
                                   llama_token *tokens, int count,
                                   char *error, size_t error_size) {
    int used = llama_memory_seq_pos_max(llama_get_memory(session->context), 0) + 1;
    if (used + count > (int)llama_n_ctx(session->context)) {
        deepseek_v4_ggml_error(error, error_size,
                               "DeepSeek V4 ggml context exhausted: %s", "increase --ctx");
        return 0;
    }
    int result = llama_decode(session->context, llama_batch_get_one(tokens, count));
    if (result != 0) {
        char detail[64];
        snprintf(detail, sizeof(detail), "llama_decode returned %d", result);
        deepseek_v4_ggml_error(error, error_size,
                               "DeepSeek V4 ggml decode failed: %s", detail);
        return 0;
    }
    return 1;
}

int deepseek_v4_ggml_generate_user(
    DeepSeekV4GgmlSession *session, const char *text, int first_turn,
    int max_new_tokens, DeepSeekV4GgmlTokenCallback callback, void *user_data,
    DeepSeekV4GgmlStats *stats, char *error, size_t error_size) {
    if (!session || !text || max_new_tokens <= 0) {
        deepseek_v4_ggml_error(error, error_size,
                               "invalid DeepSeek V4 generation request: %s",
                               "session, text, and max_new_tokens are required");
        return -1;
    }
    if (stats) *stats = {};

    std::vector<char> prompt(strlen(text) + 256);
    size_t prompt_size = deepseek_v4_chat_append_user(
        prompt.data(), prompt.size(), 0, text, first_turn);
    if (prompt_size == SIZE_MAX) {
        deepseek_v4_ggml_error(error, error_size,
                               "failed to format DeepSeek V4 prompt: %s", "buffer overflow");
        return -1;
    }
    std::vector<llama_token> prompt_tokens;
    if (!deepseek_v4_ggml_tokenize(session, prompt.data(), &prompt_tokens)) {
        deepseek_v4_ggml_error(error, error_size,
                               "failed to tokenize DeepSeek V4 prompt: %s", text);
        return -1;
    }

    llama_perf_context_reset(session->context);
    int64_t prompt_started = ggml_time_us();
    if (!deepseek_v4_ggml_decode(session, prompt_tokens.data(),
                                 (int)prompt_tokens.size(), error, error_size))
        return -1;
    double prompt_ms = (ggml_time_us() - prompt_started) / 1000.0;

    llama_sampler *sampler = llama_sampler_init_greedy();
    if (!sampler) {
        deepseek_v4_ggml_error(error, error_size,
                               "failed to initialize greedy sampler: %s", "out of memory");
        return -1;
    }
    int generated = 0;
    int64_t decode_started = ggml_time_us();
    for (; generated < max_new_tokens; generated++) {
        llama_token token = llama_sampler_sample(sampler, session->context, -1);
        if (llama_vocab_is_eog(session->vocab, token)) {
            if (!deepseek_v4_ggml_decode(session, &token, 1, error, error_size)) {
                llama_sampler_free(sampler);
                return -1;
            }
            break;
        }
        std::string piece;
        if (!deepseek_v4_ggml_piece(session, token, &piece) ||
            (callback && !callback(token, piece.data(), piece.size(), user_data))) {
            deepseek_v4_ggml_error(error, error_size,
                                   "failed to emit DeepSeek V4 token: %s", "callback rejected output");
            llama_sampler_free(sampler);
            return -1;
        }
        if (!deepseek_v4_ggml_decode(session, &token, 1, error, error_size)) {
            llama_sampler_free(sampler);
            return -1;
        }
    }
    if (generated == max_new_tokens) {
        llama_token eos = llama_vocab_eos(session->vocab);
        if (eos != LLAMA_TOKEN_NULL &&
            !deepseek_v4_ggml_decode(session, &eos, 1, error, error_size)) {
            llama_sampler_free(sampler);
            return -1;
        }
    }
    double decode_ms = (ggml_time_us() - decode_started) / 1000.0;
    llama_sampler_free(sampler);

    if (stats) {
        stats->load_ms = session->load_ms;
        stats->prompt_ms = prompt_ms;
        stats->decode_ms = decode_ms;
        stats->prompt_tokens = (int)prompt_tokens.size();
        stats->generated_tokens = generated;
    }
    if (error && error_size) error[0] = 0;
    return generated;
}

void deepseek_v4_ggml_close(DeepSeekV4GgmlSession *session) {
    if (!session) return;
    llama_free(session->context);
    llama_model_free(session->model);
    delete session;
}

#endif
