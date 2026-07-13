#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../deepseek_v4_ds4.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int touch_file(const char *path) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    fputs("GGUF", file);
    return fclose(file) == 0;
}

int main(void) {
    DeepSeekV4Ds4SpecConfig spec;
    char spec_error[256] = {0};
    unsetenv("DRAFT");
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN");
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD");
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_TAIL_CONFIDENCE_THRESHOLD");
    CHECK(deepseek_v4_ds4_spec_config_from_env(
        &spec, spec_error, sizeof(spec_error)));
    CHECK(spec.draft_tokens == 3);
    CHECK(spec.margin == 3.0f);
    CHECK(spec.confidence_threshold == 0.53f);
    CHECK(spec.tail_confidence_threshold == 0.53f);

    CHECK(setenv("DRAFT", "4", 1) == 0);
    CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN", "2.5", 1) == 0);
    CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD", "0.625", 1) == 0);
    CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_TAIL_CONFIDENCE_THRESHOLD", "0.48", 1) == 0);
    CHECK(deepseek_v4_ds4_spec_config_from_env(
        &spec, spec_error, sizeof(spec_error)));
    CHECK(spec.draft_tokens == 4);
    CHECK(spec.margin == 2.5f);
    CHECK(spec.confidence_threshold == 0.625f);
    CHECK(spec.tail_confidence_threshold == 0.48f);

    const char *bad_drafts[] = {"1", "17", "4x"};
    for (size_t i = 0; i < sizeof(bad_drafts) / sizeof(bad_drafts[0]); i++) {
        CHECK(setenv("DRAFT", bad_drafts[i], 1) == 0);
        spec_error[0] = 0;
        CHECK(!deepseek_v4_ds4_spec_config_from_env(
            &spec, spec_error, sizeof(spec_error)));
        CHECK(strstr(spec_error, "DRAFT") != NULL);
    }
    CHECK(setenv("DRAFT", "2", 1) == 0);
    const char *bad_margins[] = {"0", "-1", "nan", "2x"};
    for (size_t i = 0; i < sizeof(bad_margins) / sizeof(bad_margins[0]); i++) {
        CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN", bad_margins[i], 1) == 0);
        spec_error[0] = 0;
        CHECK(!deepseek_v4_ds4_spec_config_from_env(
            &spec, spec_error, sizeof(spec_error)));
        CHECK(strstr(spec_error, "MTP_MARGIN") != NULL);
    }
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN");
    const char *bad_thresholds[] = {"-0.1", "1.1", "nan", "0.5x"};
    for (size_t i = 0; i < sizeof(bad_thresholds) / sizeof(bad_thresholds[0]); i++) {
        CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD",
                     bad_thresholds[i], 1) == 0);
        spec_error[0] = 0;
        CHECK(!deepseek_v4_ds4_spec_config_from_env(
            &spec, spec_error, sizeof(spec_error)));
        CHECK(strstr(spec_error, "CONFIDENCE_THRESHOLD") != NULL);
    }
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD");
    for (size_t i = 0; i < sizeof(bad_thresholds) / sizeof(bad_thresholds[0]); i++) {
        CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_TAIL_CONFIDENCE_THRESHOLD",
                     bad_thresholds[i], 1) == 0);
        spec_error[0] = 0;
        CHECK(!deepseek_v4_ds4_spec_config_from_env(
            &spec, spec_error, sizeof(spec_error)));
        CHECK(strstr(spec_error, "TAIL_CONFIDENCE_THRESHOLD") != NULL);
    }
    unsetenv("DRAFT");
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN");
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD");
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_TAIL_CONFIDENCE_THRESHOLD");

    char root[] = "/tmp/floyd-dsv4-ds4.XXXXXX";
    CHECK(mkdtemp(root) != NULL);

    char checkpoint[1024], prepared[1024], model[2048], sidecar[2064];
    char support[2048], support_q4[2048], support_q4_alt[2048];
    char support_sidecar[2064];
    snprintf(checkpoint, sizeof(checkpoint), "%s/checkpoint", root);
    snprintf(prepared, sizeof(prepared), "%s-DS4", checkpoint);
    snprintf(model, sizeof(model),
             "%s/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
             prepared);
    snprintf(sidecar, sizeof(sidecar), "%s.aria2", model);
    snprintf(support, sizeof(support),
             "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-MXFP4-F16Attn.gguf",
             prepared);
    snprintf(support_q4, sizeof(support_q4),
             "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-Q4K-imatrix.gguf",
             prepared);
    snprintf(support_q4_alt, sizeof(support_q4_alt),
             "%s/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-Q4K-Q8_0-F32.gguf",
             prepared);
    snprintf(support_sidecar, sizeof(support_sidecar), "%s.part", support);
    CHECK(mkdir(checkpoint, 0700) == 0);
    CHECK(mkdir(prepared, 0700) == 0);
    CHECK(touch_file(model));
    CHECK(touch_file(sidecar));

    char found[4096], error[4096];
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_GGUF");
    CHECK(!deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                      error, sizeof(error)));
    CHECK(strstr(error, "incomplete") != NULL);
    CHECK(unlink(sidecar) == 0);
    CHECK(deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                     error, sizeof(error)));
    CHECK(strcmp(found, model) == 0);
    CHECK(strcmp(deepseek_v4_ds4_backend_name(), "metal-ds4") == 0);

    CHECK(touch_file(support));
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK");
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_MTP");
    CHECK(deepseek_v4_ds4_find_dspark_support(
        model, found, sizeof(found), error, sizeof(error)));
    CHECK(strcmp(found, support) == 0);
    CHECK(deepseek_v4_ds4_default_confidence_threshold(support) == 0.53f);
    CHECK(deepseek_v4_ds4_default_tail_confidence_threshold(support) == 0.53f);

    CHECK(touch_file(support_q4));
    CHECK(touch_file(support_q4_alt));
    CHECK(deepseek_v4_ds4_find_dspark_support(
        model, found, sizeof(found), error, sizeof(error)));
    CHECK(strcmp(found, support_q4) == 0);
    CHECK(deepseek_v4_ds4_default_confidence_threshold(support_q4) == 0.52f);
    CHECK(deepseek_v4_ds4_default_tail_confidence_threshold(support_q4) == 0.48f);
    CHECK(unlink(support_q4) == 0);
    CHECK(unlink(support_q4_alt) == 0);

    DeepSeekV4Ds4SpecKind kind = DEEPSEEK_V4_DS4_SPEC_NONE;
    CHECK(deepseek_v4_ds4_resolve_spec_model(
        model, 1, found, sizeof(found), &kind, error, sizeof(error)));
    CHECK(kind == DEEPSEEK_V4_DS4_SPEC_DSPARK);
    CHECK(strcmp(found, support) == 0);
    CHECK(deepseek_v4_ds4_resolve_spec_model(
        model, 0, found, sizeof(found), &kind, error, sizeof(error)));
    CHECK(kind == DEEPSEEK_V4_DS4_SPEC_NONE);
    CHECK(found[0] == 0);

    unsetenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK");
    CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_MTP", support, 1) == 0);
    CHECK(deepseek_v4_ds4_resolve_spec_model(
        model, 1, found, sizeof(found), &kind, error, sizeof(error)));
    CHECK(kind == DEEPSEEK_V4_DS4_SPEC_MTP);
    CHECK(strcmp(found, support) == 0);
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_MTP");

    CHECK(touch_file(support_sidecar));
    CHECK(!deepseek_v4_ds4_find_dspark_support(
        model, found, sizeof(found), error, sizeof(error)));
    CHECK(strstr(error, "incomplete") != NULL);
    CHECK(unlink(support_sidecar) == 0);
    CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK", support, 1) == 0);
    CHECK(deepseek_v4_ds4_find_dspark_support(
        model, found, sizeof(found), error, sizeof(error)));
    CHECK(strcmp(found, support) == 0);
    CHECK(unlink(support) == 0);
    CHECK(!deepseek_v4_ds4_find_dspark_support(
        model, found, sizeof(found), error, sizeof(error)));
    CHECK(strstr(error, "FLOYD_DEEPSEEK_V4_DS4_DSPARK") != NULL);
    unsetenv("FLOYD_DEEPSEEK_V4_DS4_DSPARK");

    CHECK(setenv("FLOYD_DEEPSEEK_V4_DS4_GGUF", model, 1) == 0);
    CHECK(deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                     error, sizeof(error)));
    CHECK(strcmp(found, model) == 0);

    CHECK(unlink(model) == 0);
    CHECK(!deepseek_v4_ds4_find_model(checkpoint, found, sizeof(found),
                                      error, sizeof(error)));
    CHECK(strstr(error, "FLOYD_DEEPSEEK_V4_DS4_GGUF") != NULL);

    unsetenv("FLOYD_DEEPSEEK_V4_DS4_GGUF");
    CHECK(rmdir(prepared) == 0);
    CHECK(rmdir(checkpoint) == 0);
    CHECK(rmdir(root) == 0);
    puts("DeepSeek V4 DS4 bridge contract: ok");
    return 0;
}
