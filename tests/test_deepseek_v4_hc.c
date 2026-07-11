#include <math.h>
#include <stdio.h>

#include "../deepseek_v4_hc.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define CHECK_NEAR(actual, expected, tolerance) \
    CHECK(fabsf((actual) - (expected)) <= (tolerance))

static int test_single_stream_hyper_connection(void) {
    const float fn[] = {0.0f, 0.0f, 0.0f};
    const float base[] = {0.0f, 0.0f, 0.0f};
    const float scale[] = {1.0f, 1.0f, 1.0f};
    const DeepSeekV4HyperConnection model = {
        .streams = 1, .hidden = 1, .sinkhorn_iters = 4,
        .norm_eps = 1e-6f, .hc_eps = 1e-6f,
        .fn = fn, .base = base, .scale = scale,
    };
    const float input[] = {2.0f};
    float post[1], comb[1], collapsed[1];

    CHECK(deepseek_v4_hc_forward(&model, input, 1, post, comb, collapsed));

    CHECK_NEAR(post[0], 1.0f, 1e-7f);
    CHECK_NEAR(comb[0], 1.0f / (1.0f + 1e-6f), 2e-6f);
    CHECK_NEAR(collapsed[0], (0.5f + 1e-6f) * 2.0f, 1e-6f);
    return 0;
}

static int test_comb_is_nearly_doubly_stochastic(void) {
    const float fn[32] = {
        0.1f, -0.2f, 0.3f, 0.4f,
        -0.1f, 0.2f, 0.4f, -0.3f,
        0.2f, 0.1f, -0.3f, 0.5f,
        -0.4f, 0.3f, 0.2f, 0.1f,
    };
    const float base[] = {0, 0, 0, 0, 0, 0, 0, 0};
    const float scale[] = {1, 1, 1};
    const DeepSeekV4HyperConnection model = {
        .streams = 2, .hidden = 2, .sinkhorn_iters = 8,
        .norm_eps = 1e-6f, .hc_eps = 1e-6f,
        .fn = fn, .base = base, .scale = scale,
    };
    const float input[] = {1, 2, 3, 4};
    float post[2], comb[4], collapsed[2];

    CHECK(deepseek_v4_hc_forward(&model, input, 1, post, comb, collapsed));
    CHECK_NEAR(comb[0] + comb[1], 1.0f, 2e-5f);
    CHECK_NEAR(comb[2] + comb[3], 1.0f, 2e-5f);
    CHECK_NEAR(comb[0] + comb[2], 1.0f, 2e-5f);
    CHECK_NEAR(comb[1] + comb[3], 1.0f, 2e-5f);
    return 0;
}

int main(void) {
    CHECK(test_single_stream_hyper_connection() == 0);
    CHECK(test_comb_is_nearly_doubly_stochastic() == 0);
    puts("DeepSeek V4 hyper-connection tests: ok");
    return 0;
}
