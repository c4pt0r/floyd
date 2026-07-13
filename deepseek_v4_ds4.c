#include "deepseek_v4_ds4.h"

#include <errno.h>
#include <glob.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "deepseek_v4_prefix_cache.h"

#ifdef FLOYD_DEEPSEEK_V4_DS4
#include "ds4.h"
#endif

static int ds4_regular_file(const char *path) {
    struct stat info;
    return path && stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static int ds4_copy(char *destination, size_t capacity, const char *source) {
    if (!destination || !capacity || !source) return 0;
    int count = snprintf(destination, capacity, "%s", source);
    return count >= 0 && (size_t)count < capacity;
}

static int ds4_incomplete(const char *path) {
    char sidecar[4096];
    const char *suffixes[] = {".aria2", ".part"};
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        int count = snprintf(sidecar, sizeof(sidecar), "%s%s", path, suffixes[i]);
        if (count >= 0 && (size_t)count < sizeof(sidecar) && ds4_regular_file(sidecar))
            return 1;
    }
    return 0;
}

const char *deepseek_v4_ds4_backend_name(void) {
    return "metal-ds4";
}

int deepseek_v4_ds4_prefix_cache_bytes(uint64_t megabytes, uint64_t *bytes) {
    const uint64_t bytes_per_megabyte = UINT64_C(1024) * 1024;
    if (!bytes || megabytes > UINT64_MAX / bytes_per_megabyte) return 0;
    *bytes = megabytes * bytes_per_megabyte;
    return 1;
}

int deepseek_v4_ds4_request_config_validate(
    const DeepSeekV4Ds4RequestConfig *config, char *error, size_t error_size) {
    if (!config) return 0;
    if (config->max_tokens <= 0) {
        if (error && error_size)
            snprintf(error, error_size, "max_tokens must be positive");
        return 0;
    }
    if (!isfinite(config->temperature) || config->temperature < 0.0f ||
        config->temperature > 2.0f) {
        if (error && error_size)
            snprintf(error, error_size, "temperature must be finite and in 0..2");
        return 0;
    }
    if (!isfinite(config->top_p) || config->top_p <= 0.0f ||
        config->top_p > 1.0f) {
        if (error && error_size)
            snprintf(error, error_size, "top_p must be finite and in (0,1]");
        return 0;
    }
    if (config->draft < 0 || config->draft > 16) {
        if (error && error_size)
            snprintf(error, error_size, "draft must be an integer in 0..16");
        return 0;
    }
    if (config->temperature > 0.0f && config->draft > 1) {
        if (error && error_size)
            snprintf(error, error_size,
                     "sampling with speculative draft > 1 is not supported");
        return 0;
    }
    return 1;
}

static uint64_t ds4_hash_bytes(uint64_t hash, const void *data, size_t size) {
    const unsigned char *bytes = data;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t deepseek_v4_ds4_request_config_key(
    int max_context, const DeepSeekV4Ds4RequestConfig *config) {
    if (!config) return 0;
    uint64_t hash = UINT64_C(1469598103934665603);
    uint32_t temperature_bits = 0, top_p_bits = 0;
    memcpy(&temperature_bits, &config->temperature, sizeof(temperature_bits));
    memcpy(&top_p_bits, &config->top_p, sizeof(top_p_bits));
    hash = ds4_hash_bytes(hash, &max_context, sizeof(max_context));
    hash = ds4_hash_bytes(hash, &config->draft, sizeof(config->draft));
    hash = ds4_hash_bytes(hash, &temperature_bits, sizeof(temperature_bits));
    hash = ds4_hash_bytes(hash, &top_p_bits, sizeof(top_p_bits));
    return hash;
}

float deepseek_v4_ds4_default_confidence_threshold(const char *support_path) {
    return support_path && strstr(support_path, "-Q4K") ? 0.52f : 0.53f;
}

float deepseek_v4_ds4_default_tail_confidence_threshold(
    const char *support_path) {
    return support_path && strstr(support_path, "-Q4K") ? 0.48f : 0.53f;
}

int deepseek_v4_ds4_spec_config_from_env(
    DeepSeekV4Ds4SpecConfig *config, char *error, size_t error_size) {
    if (!config) return 0;
    config->draft_tokens = 3;
    config->margin = 3.0f;
    config->confidence_threshold =
        deepseek_v4_ds4_default_confidence_threshold(NULL);
    config->tail_confidence_threshold =
        deepseek_v4_ds4_default_tail_confidence_threshold(NULL);

    const char *draft = getenv("DRAFT");
    if (draft && *draft) {
        char *end = NULL;
        errno = 0;
        long value = strtol(draft, &end, 10);
        if (errno || !end || *end || value < 2 || value > 16) {
            if (error && error_size)
                snprintf(error, error_size, "DRAFT must be an integer in 2..16");
            return 0;
        }
        config->draft_tokens = (int)value;
    }

    const char *margin = getenv("FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN");
    if (margin && *margin) {
        char *end = NULL;
        errno = 0;
        float value = strtof(margin, &end);
        if (errno || !end || *end || !isfinite(value) || value <= 0.0f) {
            if (error && error_size)
                snprintf(error, error_size,
                         "FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN must be finite and positive");
            return 0;
        }
        config->margin = value;
    }

    const char *threshold =
        getenv("FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD");
    if (threshold && *threshold) {
        char *end = NULL;
        errno = 0;
        float value = strtof(threshold, &end);
        if (errno || !end || *end || !isfinite(value) ||
            value < 0.0f || value > 1.0f) {
            if (error && error_size)
                snprintf(error, error_size,
                         "FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD must be finite and in 0..1");
            return 0;
        }
        config->confidence_threshold = value;
    }

    const char *tail_threshold =
        getenv("FLOYD_DEEPSEEK_V4_DS4_TAIL_CONFIDENCE_THRESHOLD");
    if (tail_threshold && *tail_threshold) {
        char *end = NULL;
        errno = 0;
        float value = strtof(tail_threshold, &end);
        if (errno || !end || *end || !isfinite(value) ||
            value < 0.0f || value > 1.0f) {
            if (error && error_size)
                snprintf(error, error_size,
                         "FLOYD_DEEPSEEK_V4_DS4_TAIL_CONFIDENCE_THRESHOLD must be finite and in 0..1");
            return 0;
        }
        config->tail_confidence_threshold = value;
    }
    return 1;
}

int deepseek_v4_ds4_find_model(const char *checkpoint_dir,
                               char *model_path, size_t model_path_size,
                               char *error, size_t error_size) {
    if (!checkpoint_dir || !model_path || !model_path_size) return 0;
    const char *override = getenv("FLOYD_DEEPSEEK_V4_DS4_GGUF");
    if (override && *override) {
        if (!ds4_regular_file(override)) {
            if (error && error_size)
                snprintf(error, error_size,
                         "FLOYD_DEEPSEEK_V4_DS4_GGUF is not a file: %s", override);
            return 0;
        }
        if (ds4_incomplete(override)) {
            if (error && error_size)
                snprintf(error, error_size, "DS4 GGUF download is incomplete: %s", override);
            return 0;
        }
        return ds4_copy(model_path, model_path_size, override);
    }

    const char *patterns[] = {
        "%s-DS4/DeepSeek-V4-Flash-Q4KExperts-*-imatrix.gguf",
        "%s-DS4/DeepSeek-V4-Flash-Layers37-42Q4KExperts-*-imatrix-fixed.gguf",
        "%s-DS4/DeepSeek-V4-Flash-IQ2XXS-*-imatrix.gguf",
    };
    int saw_incomplete = 0;
    for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); i++) {
        char pattern[4096];
        int count = snprintf(pattern, sizeof(pattern), patterns[i], checkpoint_dir);
        if (count < 0 || (size_t)count >= sizeof(pattern)) continue;
        glob_t matches = {0};
        int result = glob(pattern, 0, NULL, &matches);
        if (result == 0) {
            for (size_t j = 0; j < matches.gl_pathc; j++) {
                const char *candidate = matches.gl_pathv[j];
                if (!ds4_regular_file(candidate)) continue;
                if (ds4_incomplete(candidate)) {
                    saw_incomplete = 1;
                    continue;
                }
                int copied = ds4_copy(model_path, model_path_size, candidate);
                globfree(&matches);
                return copied;
            }
        }
        globfree(&matches);
    }
    if (error && error_size) {
        if (saw_incomplete)
            snprintf(error, error_size, "DS4 GGUF download is incomplete under %s-DS4",
                     checkpoint_dir);
        else
            snprintf(error, error_size, "prepared DS4 GGUF not found under %s-DS4",
                     checkpoint_dir);
    }
    return 0;
}

int deepseek_v4_ds4_find_dspark_support(
    const char *model_path, char *support_path, size_t support_path_size,
    char *error, size_t error_size) {
    if (!model_path || !support_path || !support_path_size) return 0;

    const char *override = getenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK");
    const char *override_name = "FLOYD_DEEPSEEK_V4_DS4_DSPARK";
    if (!override || !*override) {
        override = getenv("FLOYD_DEEPSEEK_V4_DS4_MTP");
        override_name = "FLOYD_DEEPSEEK_V4_DS4_MTP";
    }
    if (override && *override) {
        if (!ds4_regular_file(override)) {
            if (error && error_size)
                snprintf(error, error_size, "%s is not a file: %s",
                         override_name, override);
            return 0;
        }
        if (ds4_incomplete(override)) {
            if (error && error_size)
                snprintf(error, error_size,
                         "DSpark support GGUF download is incomplete: %s", override);
            return 0;
        }
        return ds4_copy(support_path, support_path_size, override);
    }

    char directory[4096];
    if (!ds4_copy(directory, sizeof(directory), model_path)) return 0;
    char *slash = strrchr(directory, '/');
    if (slash) *slash = 0;
    else snprintf(directory, sizeof(directory), ".");

    const char *patterns[] = {
        "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-Q4K-imatrix.gguf",
        "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-Q4K*.gguf",
        "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-MXFP4-F16Attn.gguf",
        "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-MXFP4.gguf",
    };
    int saw_incomplete = 0;
    for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); i++) {
        char pattern[4096];
        int count = snprintf(pattern, sizeof(pattern), patterns[i], directory);
        if (count < 0 || (size_t)count >= sizeof(pattern)) continue;
        glob_t matches = {0};
        int result = glob(pattern, 0, NULL, &matches);
        if (result == 0) {
            for (size_t j = 0; j < matches.gl_pathc; j++) {
                const char *candidate = matches.gl_pathv[j];
                if (strstr(candidate, "-template.gguf") ||
                    !ds4_regular_file(candidate)) continue;
                if (ds4_incomplete(candidate)) {
                    saw_incomplete = 1;
                    continue;
                }
                int copied = ds4_copy(support_path, support_path_size, candidate);
                globfree(&matches);
                return copied;
            }
        }
        globfree(&matches);
    }
    if (error && error_size) {
        if (saw_incomplete)
            snprintf(error, error_size,
                     "DSpark support GGUF download is incomplete under %s", directory);
        else
            snprintf(error, error_size,
                     "three-stage DSpark support GGUF not found under %s", directory);
    }
    return 0;
}

int deepseek_v4_ds4_resolve_spec_model(
    const char *model_path, int use_spec,
    char *support_path, size_t support_path_size,
    DeepSeekV4Ds4SpecKind *kind, char *error, size_t error_size) {
    if (!model_path || !support_path || !support_path_size || !kind) return 0;
    support_path[0] = 0;
    *kind = DEEPSEEK_V4_DS4_SPEC_NONE;
    if (!use_spec) return 1;

    const char *dspark = getenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK");
    const char *mtp = getenv("FLOYD_DEEPSEEK_V4_DS4_MTP");
    if (dspark && *dspark) {
        if (!deepseek_v4_ds4_find_dspark_support(
                model_path, support_path, support_path_size,
                error, error_size)) return 0;
        *kind = DEEPSEEK_V4_DS4_SPEC_DSPARK;
        return 1;
    }
    if (mtp && *mtp) {
        if (!deepseek_v4_ds4_find_dspark_support(
                model_path, support_path, support_path_size,
                error, error_size)) return 0;
        *kind = DEEPSEEK_V4_DS4_SPEC_MTP;
        return 1;
    }
    if (deepseek_v4_ds4_find_dspark_support(
            model_path, support_path, support_path_size,
            error, error_size)) {
        *kind = DEEPSEEK_V4_DS4_SPEC_DSPARK;
    } else if (error && error_size) {
        error[0] = 0;
    }
    return 1;
}

#ifdef FLOYD_DEEPSEEK_V4_DS4

struct DeepSeekV4Ds4Session {
    ds4_engine *engine;
    ds4_session *session;
    ds4_tokens transcript;
    int max_context;
    int use_spec;
    int draft_tokens;
    double load_ms;
    char *fingerprint;
    DeepSeekV4PrefixCache prefix_cache;
    uint64_t rng;
};

static double ds4_now_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

static int ds4_configure_metal_sources(char *error, size_t error_size) {
#ifdef FLOYD_DS4_SOURCE_DIR
    static const struct {
        const char *environment;
        const char *file;
    } sources[] = {
        {"DS4_METAL_FLASH_ATTN_SOURCE", "flash_attn.metal"},
        {"DS4_METAL_DENSE_SOURCE", "dense.metal"},
        {"DS4_METAL_MOE_SOURCE", "moe.metal"},
        {"DS4_METAL_DSV4_HC_SOURCE", "dsv4_hc.metal"},
        {"DS4_METAL_UNARY_SOURCE", "unary.metal"},
        {"DS4_METAL_DSV4_KV_SOURCE", "dsv4_kv.metal"},
        {"DS4_METAL_DSV4_ROPE_SOURCE", "dsv4_rope.metal"},
        {"DS4_METAL_DSV4_MISC_SOURCE", "dsv4_misc.metal"},
        {"DS4_METAL_ARGSORT_SOURCE", "argsort.metal"},
        {"DS4_METAL_CPY_SOURCE", "cpy.metal"},
        {"DS4_METAL_CONCAT_SOURCE", "concat.metal"},
        {"DS4_METAL_GET_ROWS_SOURCE", "get_rows.metal"},
        {"DS4_METAL_SUM_ROWS_SOURCE", "sum_rows.metal"},
        {"DS4_METAL_SOFTMAX_SOURCE", "softmax.metal"},
        {"DS4_METAL_REPEAT_SOURCE", "repeat.metal"},
        {"DS4_METAL_GLU_SOURCE", "glu.metal"},
        {"DS4_METAL_NORM_SOURCE", "norm.metal"},
        {"DS4_METAL_BIN_SOURCE", "bin.metal"},
        {"DS4_METAL_SET_ROWS_SOURCE", "set_rows.metal"},
    };
    for (size_t i = 0; i < sizeof(sources) / sizeof(sources[0]); i++) {
        if (getenv(sources[i].environment)) continue;
        char path[4096];
        int count = snprintf(path, sizeof(path), "%s/metal/%s",
                             FLOYD_DS4_SOURCE_DIR, sources[i].file);
        if (count < 0 || (size_t)count >= sizeof(path) ||
            setenv(sources[i].environment, path, 0) != 0) {
            if (error && error_size)
                snprintf(error, error_size, "failed to configure DS4 Metal sources");
            return 0;
        }
    }
#else
    (void)error;
    (void)error_size;
#endif
    return 1;
}

static char *ds4_model_fingerprint(
    const char *model_path, const char *support_path) {
    char model_real[4096], support_real[4096];
    const char *model = realpath(model_path, model_real) ? model_real : model_path;
    const char *support = "none";
    if (support_path && *support_path)
        support = realpath(support_path, support_real) ? support_real : support_path;
    struct stat model_stat = {0}, support_stat = {0};
    if (stat(model_path, &model_stat) != 0) return NULL;
    if (support_path && *support_path && stat(support_path, &support_stat) != 0)
        return NULL;
    int needed = snprintf(
        NULL, 0, "ds4-payload-v%u|%s|%lld|%lld|%s|%lld|%lld",
        DS4_SESSION_PAYLOAD_VERSION, model,
        (long long)model_stat.st_size, (long long)model_stat.st_mtime,
        support, (long long)support_stat.st_size,
        (long long)support_stat.st_mtime);
    if (needed < 0) return NULL;
    char *fingerprint = malloc((size_t)needed + 1);
    if (!fingerprint) return NULL;
    snprintf(fingerprint, (size_t)needed + 1,
             "ds4-payload-v%u|%s|%lld|%lld|%s|%lld|%lld",
             DS4_SESSION_PAYLOAD_VERSION, model,
             (long long)model_stat.st_size, (long long)model_stat.st_mtime,
             support, (long long)support_stat.st_size,
             (long long)support_stat.st_mtime);
    return fingerprint;
}

DeepSeekV4Ds4Session *deepseek_v4_ds4_open_cached(
    const char *model_path, int max_context, int use_spec,
    uint64_t prefix_cache_bytes, char *error, size_t error_size) {
    if (!model_path || max_context <= 0) return NULL;
    if (!ds4_configure_metal_sources(error, error_size)) return NULL;
    DeepSeekV4Ds4Session *result = calloc(1, sizeof(*result));
    if (!result) return NULL;
    char support_path[4096];
    DeepSeekV4Ds4SpecKind spec_kind;
    if (!deepseek_v4_ds4_resolve_spec_model(
            model_path, use_spec, support_path, sizeof(support_path),
            &spec_kind, error, error_size)) {
        free(result);
        return NULL;
    }
    const char *mtp_path = spec_kind == DEEPSEEK_V4_DS4_SPEC_MTP
        ? support_path : NULL;
    const char *dspark_path = spec_kind == DEEPSEEK_V4_DS4_SPEC_DSPARK
        ? support_path : NULL;
    result->fingerprint = ds4_model_fingerprint(model_path, support_path);
    if (!result->fingerprint) {
        if (error && error_size)
            snprintf(error, error_size, "failed to fingerprint DS4 model files");
        free(result);
        return NULL;
    }
    deepseek_v4_prefix_cache_init(&result->prefix_cache, prefix_cache_bytes);
    DeepSeekV4Ds4SpecConfig spec = {
        .draft_tokens = 1,
        .margin = 3.0f,
        .confidence_threshold = 0.5f,
        .tail_confidence_threshold = 0.5f,
    };
    if (spec_kind != DEEPSEEK_V4_DS4_SPEC_NONE &&
        !deepseek_v4_ds4_spec_config_from_env(&spec, error, error_size)) {
        deepseek_v4_prefix_cache_free(&result->prefix_cache);
        free(result->fingerprint);
        free(result);
        return NULL;
    }
    const char *threshold_override =
        getenv("FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD");
    if (spec_kind == DEEPSEEK_V4_DS4_SPEC_DSPARK &&
        (!threshold_override || !*threshold_override)) {
        spec.confidence_threshold =
            deepseek_v4_ds4_default_confidence_threshold(dspark_path);
    }
    const char *tail_threshold_override =
        getenv("FLOYD_DEEPSEEK_V4_DS4_TAIL_CONFIDENCE_THRESHOLD");
    if (spec_kind == DEEPSEEK_V4_DS4_SPEC_DSPARK &&
        (!tail_threshold_override || !*tail_threshold_override)) {
        spec.tail_confidence_threshold =
            deepseek_v4_ds4_default_tail_confidence_threshold(dspark_path);
    }
    ds4_engine_options options = {
        .model_path = model_path,
        .mtp_path = mtp_path,
        .dspark_path = dspark_path,
        .backend = DS4_BACKEND_METAL,
        .warm_weights = true,
        .mtp_draft_tokens = spec.draft_tokens,
        .mtp_margin = spec.margin,
        .dspark_confidence_threshold = spec.confidence_threshold,
        .dspark_confidence_tail_threshold = spec.tail_confidence_threshold,
    };
    double started = ds4_now_ms();
    if (ds4_engine_open(&result->engine, &options) != 0 ||
        ds4_session_create(&result->session, result->engine, max_context) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "failed to initialize resident DS4 Metal runtime");
        if (result->engine) ds4_engine_close(result->engine);
        deepseek_v4_prefix_cache_free(&result->prefix_cache);
        free(result->fingerprint);
        free(result);
        return NULL;
    }
    ds4_chat_begin(result->engine, &result->transcript);
    result->max_context = max_context;
    result->use_spec = ds4_engine_mtp_draft_tokens(result->engine) > 1;
    result->draft_tokens = ds4_engine_mtp_draft_tokens(result->engine);
    result->load_ms = ds4_now_ms() - started;
    result->rng = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)result;
    if (use_spec) {
        if (!result->use_spec) {
            fprintf(stderr, "DEEPSEEK_V4_SPEC disabled=dspark-not-prepared\n");
        } else {
            const char *backend = spec_kind == DEEPSEEK_V4_DS4_SPEC_DSPARK
                ? "dspark" : "mtp";
            fprintf(stderr,
                    "DEEPSEEK_V4_SPEC backend=%s draft=%d confidence_threshold=%.3f "
                    "tail_confidence_threshold=%.3f\n",
                    backend, spec.draft_tokens, spec.confidence_threshold,
                    spec.tail_confidence_threshold);
        }
    }
    return result;
}

DeepSeekV4Ds4Session *deepseek_v4_ds4_open(
    const char *model_path, int max_context, int use_spec,
    char *error, size_t error_size) {
    return deepseek_v4_ds4_open_cached(
        model_path, max_context, use_spec, 0, error, error_size);
}

int deepseek_v4_ds4_reset(DeepSeekV4Ds4Session *session) {
    if (!session || !session->engine) return 0;
    ds4_session *replacement = NULL;
    if (ds4_session_create(&replacement, session->engine, session->max_context) != 0)
        return 0;
    ds4_session_free(session->session);
    session->session = replacement;
    ds4_tokens_free(&session->transcript);
    ds4_chat_begin(session->engine, &session->transcript);
    return 1;
}

int deepseek_v4_ds4_generate_user(
    DeepSeekV4Ds4Session *session, const char *text, int max_new_tokens,
    DeepSeekV4Ds4TokenCallback callback, void *user_data,
    DeepSeekV4Ds4Stats *stats, char *error, size_t error_size) {
    if (!session || !text || max_new_tokens <= 0) return -1;
    if (stats) memset(stats, 0, sizeof(*stats));

    int rollback = session->transcript.len;
    ds4_chat_append_message(session->engine, &session->transcript, "user", text);
    ds4_chat_append_assistant_prefix(session->engine, &session->transcript,
                                     DS4_THINK_NONE);
    int old_pos = ds4_session_pos(session->session);
    int common = ds4_session_common_prefix(session->session, &session->transcript);
    int cached = common == old_pos && session->transcript.len >= old_pos ? common : 0;
    int suffix = session->transcript.len - cached;
    ds4_session_spec_stats spec_before = {0}, spec_after = {0};
    (void)ds4_session_get_spec_stats(session->session, &spec_before);
    double prompt_started = ds4_now_ms();
    if (ds4_session_sync(session->session, &session->transcript,
                         error, error_size) != 0) {
        session->transcript.len = rollback;
        return -1;
    }
    double prompt_ms = ds4_now_ms() - prompt_started;

    int generated = 0, spec_rounds = 0, spec_tokens = 0;
    double decode_started = ds4_now_ms();
    while (generated < max_new_tokens) {
        int token = ds4_session_argmax(session->session);
        if (token < 0) return -1;
        int tokens[17];
        int count;
        if (session->use_spec) {
            count = ds4_session_eval_speculative_argmax(
                session->session, token, max_new_tokens - generated,
                ds4_token_eos(session->engine), tokens, 17, error, error_size);
            if (count < 0) return -1;
            spec_rounds++;
            if (count > 1) spec_tokens += count - 1;
        } else {
            if (ds4_session_eval(session->session, token, error, error_size) != 0)
                return -1;
            tokens[0] = token;
            count = 1;
        }
        int stop = 0;
        for (int i = 0; i < count && generated < max_new_tokens; i++) {
            token = tokens[i];
            ds4_tokens_push(&session->transcript, token);
            if (token == ds4_token_eos(session->engine)) {
                stop = 1;
                break;
            }
            size_t piece_size = 0;
            char *piece = ds4_token_text(session->engine, token, &piece_size);
            int accepted = piece && (!callback || callback(
                token, piece, piece_size, user_data));
            free(piece);
            if (!accepted) return -1;
            generated++;
        }
        if (stop) break;
    }
    if (generated == max_new_tokens)
        ds4_tokens_push(&session->transcript, ds4_token_eos(session->engine));
    double decode_ms = ds4_now_ms() - decode_started;

    if (stats) {
        stats->load_ms = session->load_ms;
        stats->prompt_ms = prompt_ms;
        stats->decode_ms = decode_ms;
        stats->prompt_tokens = suffix;
        stats->generated_tokens = generated;
        stats->speculative_rounds = spec_rounds;
        stats->speculative_tokens = spec_tokens;
        if (ds4_session_get_spec_stats(session->session, &spec_after)) {
            stats->speculative_proposed =
                (int)(spec_after.proposed_tokens - spec_before.proposed_tokens);
            stats->speculative_direct_accepted =
                (int)(spec_after.direct_accepted_tokens -
                      spec_before.direct_accepted_tokens);
            stats->speculative_prefix1_accepted =
                (int)(spec_after.prefix1_accepted_tokens -
                      spec_before.prefix1_accepted_tokens);
            stats->speculative_frontier_snapshots =
                (int)(spec_after.frontier_snapshots -
                      spec_before.frontier_snapshots);
            stats->speculative_proposal_early_skips =
                (int)(spec_after.proposal_early_skips -
                      spec_before.proposal_early_skips);
            stats->speculative_target_ms =
                spec_after.target_ms - spec_before.target_ms;
            stats->speculative_proposal_ms =
                spec_after.proposal_ms - spec_before.proposal_ms;
            stats->speculative_verify_ms =
                spec_after.verify_ms - spec_before.verify_ms;
            stats->speculative_verify_layer_encode_ms =
                spec_after.verify_layer_encode_ms -
                spec_before.verify_layer_encode_ms;
            stats->speculative_verify_layer_execute_ms =
                spec_after.verify_layer_execute_ms -
                spec_before.verify_layer_execute_ms;
            stats->speculative_verify_head_ms =
                spec_after.verify_head_ms - spec_before.verify_head_ms;
            stats->speculative_verify_read_ms =
                spec_after.verify_read_ms - spec_before.verify_read_ms;
            stats->speculative_replay_ms =
                spec_after.replay_ms - spec_before.replay_ms;
            for (int draft = 0; draft <= 6; draft++) {
                for (int commit = 0; commit <= 6; commit++) {
                    stats->speculative_verify_outcome_calls[draft][commit] =
                        spec_after.verify_outcome_calls[draft][commit] -
                        spec_before.verify_outcome_calls[draft][commit];
                    stats->speculative_verify_outcome_ms[draft][commit] =
                        spec_after.verify_outcome_ms[draft][commit] -
                        spec_before.verify_outcome_ms[draft][commit];
                }
            }
        }
    }
    return generated;
}

static uint64_t ds4_available_memory_bytes(void) {
#ifdef __APPLE__
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    vm_statistics64_data_t stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_page_size(host, &page_size) != KERN_SUCCESS ||
        host_statistics64(host, HOST_VM_INFO64,
                          (host_info64_t)&stats, &count) != KERN_SUCCESS)
        return UINT64_MAX;
    uint64_t pages = (uint64_t)stats.free_count + stats.inactive_count +
                     stats.purgeable_count;
    return pages > UINT64_MAX / page_size ? UINT64_MAX : pages * page_size;
#elif defined(__linux__)
    FILE *file = fopen("/proc/meminfo", "r");
    if (!file) return UINT64_MAX;
    char line[256];
    unsigned long long kilobytes = 0;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kilobytes) == 1) break;
    }
    fclose(file);
    return kilobytes > UINT64_MAX / 1024 ? UINT64_MAX : kilobytes * 1024;
#else
    return UINT64_MAX;
#endif
}

static void ds4_cache_save(
    DeepSeekV4Ds4Session *session, const ds4_tokens *tokens,
    uint64_t config_key) {
    if (!session || !tokens || tokens->len <= 0 ||
        session->prefix_cache.budget_bytes == 0) return;
    uint64_t payload_bytes = ds4_session_payload_bytes(session->session);
    uint64_t entry_bytes = deepseek_v4_prefix_cache_entry_bytes(
        session->fingerprint, (size_t)tokens->len, payload_bytes);
    if (!deepseek_v4_prefix_cache_prepare_insert(
            &session->prefix_cache, entry_bytes,
            ds4_available_memory_bytes())) return;

    ds4_session_snapshot snapshot = {0};
    char cache_error[256] = {0};
    if (ds4_session_save_snapshot(
            session->session, &snapshot,
            cache_error, sizeof(cache_error)) != 0) {
        ds4_session_snapshot_free(&snapshot);
        return;
    }
    if (deepseek_v4_prefix_cache_put_take(
            &session->prefix_cache, session->fingerprint, config_key,
            tokens->v, (size_t)tokens->len, snapshot.ptr, snapshot.len)) {
        snapshot.ptr = NULL;
        snapshot.len = snapshot.cap = 0;
    }
    ds4_session_snapshot_free(&snapshot);
}

static int ds4_messages_validate(
    const DeepSeekV4Ds4Message *messages, size_t message_count,
    char *error, size_t error_size) {
    if (!messages || message_count == 0) {
        if (error && error_size) snprintf(error, error_size, "messages must not be empty");
        return 0;
    }
    for (size_t i = 0; i < message_count; i++) {
        const char *role = messages[i].role;
        if (!role || !messages[i].content ||
            (strcmp(role, "system") && strcmp(role, "user") &&
             strcmp(role, "assistant"))) {
            if (error && error_size)
                snprintf(error, error_size, "messages contain an unsupported role");
            return 0;
        }
        if (!strcmp(role, "system") && i != 0) {
            if (error && error_size)
                snprintf(error, error_size, "system message must be first");
            return 0;
        }
    }
    if (strcmp(messages[message_count - 1].role, "user") != 0) {
        if (error && error_size)
            snprintf(error, error_size, "last message must have role user");
        return 0;
    }
    return 1;
}

static void ds4_stats_add_spec_delta(
    DeepSeekV4Ds4Stats *stats,
    const ds4_session_spec_stats *before,
    const ds4_session_spec_stats *after) {
    if (!stats || !before || !after) return;
    stats->speculative_proposed =
        (int)(after->proposed_tokens - before->proposed_tokens);
    stats->speculative_direct_accepted =
        (int)(after->direct_accepted_tokens - before->direct_accepted_tokens);
    stats->speculative_prefix1_accepted =
        (int)(after->prefix1_accepted_tokens - before->prefix1_accepted_tokens);
    stats->speculative_frontier_snapshots =
        (int)(after->frontier_snapshots - before->frontier_snapshots);
    stats->speculative_proposal_early_skips =
        (int)(after->proposal_early_skips - before->proposal_early_skips);
    stats->speculative_target_ms = after->target_ms - before->target_ms;
    stats->speculative_proposal_ms = after->proposal_ms - before->proposal_ms;
    stats->speculative_verify_ms = after->verify_ms - before->verify_ms;
    stats->speculative_verify_layer_encode_ms =
        after->verify_layer_encode_ms - before->verify_layer_encode_ms;
    stats->speculative_verify_layer_execute_ms =
        after->verify_layer_execute_ms - before->verify_layer_execute_ms;
    stats->speculative_verify_head_ms =
        after->verify_head_ms - before->verify_head_ms;
    stats->speculative_verify_read_ms =
        after->verify_read_ms - before->verify_read_ms;
    stats->speculative_replay_ms = after->replay_ms - before->replay_ms;
    for (int draft = 0; draft <= 6; draft++) {
        for (int commit = 0; commit <= 6; commit++) {
            stats->speculative_verify_outcome_calls[draft][commit] =
                after->verify_outcome_calls[draft][commit] -
                before->verify_outcome_calls[draft][commit];
            stats->speculative_verify_outcome_ms[draft][commit] =
                after->verify_outcome_ms[draft][commit] -
                before->verify_outcome_ms[draft][commit];
        }
    }
}

int deepseek_v4_ds4_generate_messages(
    DeepSeekV4Ds4Session *session,
    const DeepSeekV4Ds4Message *messages, size_t message_count,
    const DeepSeekV4Ds4RequestConfig *config,
    DeepSeekV4Ds4TokenCallback callback, void *user_data,
    DeepSeekV4Ds4Stats *stats, char *error, size_t error_size) {
    if (!session ||
        !deepseek_v4_ds4_request_config_validate(config, error, error_size) ||
        !ds4_messages_validate(messages, message_count, error, error_size))
        return -1;
    if (config->draft > 1 && !session->use_spec) {
        if (error && error_size)
            snprintf(error, error_size, "speculative draft requested without DSpark support");
        return -1;
    }
    if (config->draft > 1 && config->draft != session->draft_tokens) {
        if (error && error_size)
            snprintf(error, error_size,
                     "request draft %d differs from resident DS4 draft %d",
                     config->draft, session->draft_tokens);
        return -1;
    }
    if (stats) memset(stats, 0, sizeof(*stats));

    ds4_tokens prompt = {0};
    ds4_chat_begin(session->engine, &prompt);
    for (size_t i = 0; i + 1 < message_count; i++)
        ds4_chat_append_message(
            session->engine, &prompt, messages[i].role, messages[i].content);
    int anchor_tokens = prompt.len;
    ds4_chat_append_message(
        session->engine, &prompt, messages[message_count - 1].role,
        messages[message_count - 1].content);
    ds4_chat_append_assistant_prefix(
        session->engine, &prompt, DS4_THINK_NONE);
    if (prompt.len >= session->max_context) {
        if (error && error_size)
            snprintf(error, error_size, "prompt exceeds DS4 context");
        ds4_tokens_free(&prompt);
        return -1;
    }

    uint64_t config_key = deepseek_v4_ds4_request_config_key(
        session->max_context, config);
    int cached_tokens = 0;
    double prompt_started = ds4_now_ms();
    if (session->prefix_cache.budget_bytes > 0) {
        DeepSeekV4PrefixCacheHit hit = {0};
        if (deepseek_v4_prefix_cache_find_longest(
                &session->prefix_cache, session->fingerprint, config_key,
                prompt.v, (size_t)prompt.len, &hit)) {
            ds4_session_snapshot snapshot = {
                .ptr = (uint8_t *)hit.snapshot,
                .len = hit.snapshot_bytes,
                .cap = hit.snapshot_bytes,
            };
            char cache_error[256] = {0};
            if (ds4_session_load_snapshot(
                    session->session, &snapshot,
                    cache_error, sizeof(cache_error)) == 0)
                cached_tokens = hit.prefix_tokens;
        }
    }
    if (cached_tokens == 0 && !deepseek_v4_ds4_reset(session)) {
        if (error && error_size) snprintf(error, error_size, "failed to reset DS4 session");
        ds4_tokens_free(&prompt);
        return -1;
    }

    if (message_count > 1 && cached_tokens < anchor_tokens) {
        int full_tokens = prompt.len;
        prompt.len = anchor_tokens;
        if (ds4_session_sync(session->session, &prompt, error, error_size) != 0) {
            prompt.len = full_tokens;
            ds4_tokens_free(&prompt);
            return -1;
        }
        ds4_cache_save(session, &prompt, config_key);
        prompt.len = full_tokens;
    }
    if (ds4_session_sync(session->session, &prompt, error, error_size) != 0) {
        ds4_tokens_free(&prompt);
        return -1;
    }
    if (cached_tokens < prompt.len) ds4_cache_save(session, &prompt, config_key);
    double prompt_ms = ds4_now_ms() - prompt_started;

    ds4_session_spec_stats spec_before = {0}, spec_after = {0};
    (void)ds4_session_get_spec_stats(session->session, &spec_before);
    int generated = 0, spec_rounds = 0, spec_tokens = 0;
    double decode_started = ds4_now_ms();
    while (generated < config->max_tokens) {
        int token = config->temperature == 0.0f
            ? ds4_session_argmax(session->session)
            : ds4_session_sample(session->session, config->temperature, 0,
                                 config->top_p, 0.0f, &session->rng);
        if (token < 0) {
            if (error && error_size) snprintf(error, error_size, "DS4 sampling failed");
            generated = -1;
            break;
        }
        int tokens[17], count;
        if (config->draft > 1) {
            count = ds4_session_eval_speculative_argmax(
                session->session, token, config->max_tokens - generated,
                ds4_token_eos(session->engine), tokens, 17, error, error_size);
            if (count < 0) { generated = -1; break; }
            spec_rounds++;
            if (count > 1) spec_tokens += count - 1;
        } else {
            if (ds4_session_eval(session->session, token, error, error_size) != 0) {
                generated = -1;
                break;
            }
            tokens[0] = token;
            count = 1;
        }
        int stop = 0;
        for (int i = 0; i < count && generated < config->max_tokens; i++) {
            token = tokens[i];
            if (token == ds4_token_eos(session->engine)) { stop = 1; break; }
            size_t piece_size = 0;
            char *piece = ds4_token_text(session->engine, token, &piece_size);
            int accepted = piece && (!callback || callback(
                token, piece, piece_size, user_data));
            free(piece);
            if (!accepted) { generated = -1; stop = 1; break; }
            generated++;
        }
        if (stop) break;
    }
    double decode_ms = ds4_now_ms() - decode_started;

    if (stats) {
        DeepSeekV4PrefixCacheStats cache_stats =
            deepseek_v4_prefix_cache_stats(&session->prefix_cache);
        stats->load_ms = session->load_ms;
        stats->prompt_ms = prompt_ms;
        stats->decode_ms = decode_ms;
        stats->prompt_tokens = prompt.len - cached_tokens;
        stats->cached_tokens = cached_tokens;
        stats->generated_tokens = generated < 0 ? 0 : generated;
        stats->cache_hit = cached_tokens > 0;
        stats->cache_prefix_tokens = cached_tokens;
        stats->cache_entries = cache_stats.entries;
        stats->cache_bytes = cache_stats.bytes;
        stats->speculative_rounds = spec_rounds;
        stats->speculative_tokens = spec_tokens;
        if (ds4_session_get_spec_stats(session->session, &spec_after))
            ds4_stats_add_spec_delta(stats, &spec_before, &spec_after);
    }
    ds4_tokens_free(&prompt);
    return generated;
}

void deepseek_v4_ds4_close(DeepSeekV4Ds4Session *session) {
    if (!session) return;
    ds4_session_free(session->session);
    ds4_tokens_free(&session->transcript);
    ds4_engine_close(session->engine);
    deepseek_v4_prefix_cache_free(&session->prefix_cache);
    free(session->fingerprint);
    free(session);
}

#endif
