#include "deepseek_v4_ds4.h"

#include <errno.h>
#include <glob.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

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

int deepseek_v4_ds4_spec_config_from_env(
    DeepSeekV4Ds4SpecConfig *config, char *error, size_t error_size) {
    if (!config) return 0;
    config->draft_tokens = 3;
    config->margin = 3.0f;

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
        "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-MXFP4-F16Attn.gguf",
        "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-MXFP4.gguf",
        "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-Q4K*.gguf",
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
    double load_ms;
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

DeepSeekV4Ds4Session *deepseek_v4_ds4_open(
    const char *model_path, int max_context, int use_spec,
    char *error, size_t error_size) {
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
    DeepSeekV4Ds4SpecConfig spec = {.draft_tokens = 1, .margin = 3.0f};
    if (spec_kind != DEEPSEEK_V4_DS4_SPEC_NONE &&
        !deepseek_v4_ds4_spec_config_from_env(&spec, error, error_size)) {
        free(result);
        return NULL;
    }
    ds4_engine_options options = {
        .model_path = model_path,
        .mtp_path = mtp_path,
        .dspark_path = dspark_path,
        .backend = DS4_BACKEND_METAL,
        .warm_weights = true,
        .mtp_draft_tokens = spec.draft_tokens,
        .mtp_margin = spec.margin,
    };
    double started = ds4_now_ms();
    if (ds4_engine_open(&result->engine, &options) != 0 ||
        ds4_session_create(&result->session, result->engine, max_context) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "failed to initialize resident DS4 Metal runtime");
        if (result->engine) ds4_engine_close(result->engine);
        free(result);
        return NULL;
    }
    ds4_chat_begin(result->engine, &result->transcript);
    result->max_context = max_context;
    result->use_spec = ds4_engine_mtp_draft_tokens(result->engine) > 1;
    result->load_ms = ds4_now_ms() - started;
    if (use_spec) {
        if (!result->use_spec) {
            fprintf(stderr, "DEEPSEEK_V4_SPEC disabled=dspark-not-prepared\n");
        } else {
            const char *backend = spec_kind == DEEPSEEK_V4_DS4_SPEC_DSPARK
                ? "dspark" : "mtp";
            fprintf(stderr, "DEEPSEEK_V4_SPEC backend=%s draft=%d\n",
                    backend, spec.draft_tokens);
        }
    }
    return result;
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
            stats->speculative_target_ms =
                spec_after.target_ms - spec_before.target_ms;
            stats->speculative_proposal_ms =
                spec_after.proposal_ms - spec_before.proposal_ms;
            stats->speculative_verify_ms =
                spec_after.verify_ms - spec_before.verify_ms;
            stats->speculative_replay_ms =
                spec_after.replay_ms - spec_before.replay_ms;
        }
    }
    return generated;
}

void deepseek_v4_ds4_close(DeepSeekV4Ds4Session *session) {
    if (!session) return;
    ds4_session_free(session->session);
    ds4_tokens_free(&session->transcript);
    ds4_engine_close(session->engine);
    free(session);
}

#endif
