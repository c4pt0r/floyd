#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../st.h"
#include "../v4_hc.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float *load_f32(shards *source, const char *name, int64_t expected) {
    int64_t count = st_numel(source, name);
    if (count != expected) {
        fprintf(stderr, "%s: elements=%lld expected=%lld\n", name,
                (long long)count, (long long)expected);
        exit(1);
    }
    float *data = malloc((size_t)count * sizeof(float));
    st_read_f32(source, name, data, 0);
    return data;
}

static float max_error(const float *actual, const float *expected, int64_t count) {
    float maximum = 0.0f;
    for (int64_t i = 0; i < count; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > maximum) maximum = error;
    }
    return maximum;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_tiny_v4>\n", argv[0]);
        return 2;
    }

    enum { H = 2, D = 64, MIX = (2 + H) * H };
    shards source;
    st_init(&source, argv[1]);
    int64_t n_tokens = st_numel(&source, "hc.0.attn.post") / H;
    CHECK(n_tokens > 0);

    for (int layer = 0; layer < 3; layer++) {
        for (int site_index = 0; site_index < 2; site_index++) {
            const char *site = site_index == 0 ? "attn" : "ffn";
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s_hc.fn", layer, site);
            float *fn = load_f32(&source, name, MIX * H * D);
            snprintf(name, sizeof(name), "model.layers.%d.%s_hc.base", layer, site);
            float *base = load_f32(&source, name, MIX);
            snprintf(name, sizeof(name), "model.layers.%d.%s_hc.scale", layer, site);
            float *scale = load_f32(&source, name, 3);

            snprintf(name, sizeof(name), "hc.%d.%s.input", layer, site);
            float *input = load_f32(&source, name, n_tokens * H * D);
            snprintf(name, sizeof(name), "hc.%d.%s.post", layer, site);
            float *expected_post = load_f32(&source, name, n_tokens * H);
            snprintf(name, sizeof(name), "hc.%d.%s.comb", layer, site);
            float *expected_comb = load_f32(&source, name, n_tokens * H * H);
            snprintf(name, sizeof(name), "hc.%d.%s.collapsed", layer, site);
            float *expected_collapsed = load_f32(&source, name, n_tokens * D);

            float *post = malloc((size_t)n_tokens * H * sizeof(float));
            float *comb = malloc((size_t)n_tokens * H * H * sizeof(float));
            float *collapsed = malloc((size_t)n_tokens * D * sizeof(float));
            V4HyperConnection model = {
                .streams = H, .hidden = D, .sinkhorn_iters = 4,
                .norm_eps = 1e-6f, .hc_eps = 1e-6f,
                .fn = fn, .base = base, .scale = scale,
            };
            CHECK(v4_hc_forward(&model, input, (int)n_tokens, post, comb, collapsed));

            float post_error = max_error(post, expected_post, n_tokens * H);
            float comb_error = max_error(comb, expected_comb, n_tokens * H * H);
            float collapsed_error = max_error(collapsed, expected_collapsed, n_tokens * D);
            printf("v4 hc layer %d %s: post=%.9g comb=%.9g collapsed=%.9g\n",
                   layer, site, post_error, comb_error, collapsed_error);
            CHECK(post_error < 2e-5f);
            CHECK(comb_error < 2e-5f);
            CHECK(collapsed_error < 2e-5f);
        }
    }
    puts("v4 hyper-connection fixture tests: ok");
    return 0;
}
