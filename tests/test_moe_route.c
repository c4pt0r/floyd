#include <math.h>
#include <stdio.h>

#include "../moe_route.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define CHECK_NEAR(actual, expected, tolerance) \
    CHECK(fabsf((actual) - (expected)) <= (tolerance))

static int test_sigmoid_bias_affects_selection_only(void) {
    const float logits[] = {0.0f, 2.0f, -2.0f};
    const float bias[] = {0.0f, -1.5f, 2.0f};
    int indices[2];
    float weights[2], scores[3], choices[3];

    int n = moe_route_select(logits, bias, 3, 2, MOE_SCORE_SIGMOID,
                             indices, weights, scores, choices);

    CHECK(n == 2);
    CHECK(indices[0] == 2);
    CHECK(indices[1] == 0);
    CHECK_NEAR(weights[0], 1.0f / (1.0f + expf(2.0f)), 1e-6f);
    CHECK_NEAR(weights[1], 0.5f, 1e-6f);
    CHECK_NEAR(choices[0], scores[0], 1e-6f);
    CHECK_NEAR(choices[2], scores[2] + 2.0f, 1e-6f);
    return 0;
}

static int test_softmax_is_stable_and_selects_top_k(void) {
    const float logits[] = {1000.0f, 1001.0f, 999.0f};
    const float denom = expf(-1.0f) + 1.0f + expf(-2.0f);
    int indices[2];
    float weights[2], scores[3], choices[3];

    int n = moe_route_select(logits, NULL, 3, 2, MOE_SCORE_SOFTMAX,
                             indices, weights, scores, choices);

    CHECK(n == 2);
    CHECK(indices[0] == 1);
    CHECK(indices[1] == 0);
    CHECK_NEAR(weights[0], 1.0f / denom, 1e-6f);
    CHECK_NEAR(weights[1], expf(-1.0f) / denom, 1e-6f);
    CHECK_NEAR(scores[0] + scores[1] + scores[2], 1.0f, 1e-6f);
    return 0;
}

static int test_sqrt_softplus_scores(void) {
    const float logits[] = {-2.0f, 0.0f, 2.0f};
    int indices[2];
    float weights[2], scores[3], choices[3];

    int n = moe_route_select(logits, NULL, 3, 2, MOE_SCORE_SQRT_SOFTPLUS,
                             indices, weights, scores, choices);

    CHECK(n == 2);
    CHECK(indices[0] == 2);
    CHECK(indices[1] == 1);
    CHECK_NEAR(weights[0], sqrtf(log1pf(expf(2.0f))), 1e-6f);
    CHECK_NEAR(weights[1], sqrtf(logf(2.0f)), 1e-6f);
    return 0;
}

static int test_ties_keep_lower_expert_index(void) {
    const float logits[] = {0.0f, 0.0f, 0.0f};
    int indices[2];
    float weights[2], scores[3], choices[3];

    moe_route_select(logits, NULL, 3, 2, MOE_SCORE_SIGMOID,
                     indices, weights, scores, choices);

    CHECK(indices[0] == 0);
    CHECK(indices[1] == 1);
    return 0;
}

static int test_finalize_uses_effective_expert_count(void) {
    float weights[] = {0.2f, 0.3f, 0.5f};

    moe_route_finalize(weights, 2, 1, 2.0f);

    CHECK_NEAR(weights[0], 0.8f, 1e-6f);
    CHECK_NEAR(weights[1], 1.2f, 1e-6f);
    CHECK_NEAR(weights[2], 0.5f, 1e-6f);
    return 0;
}

int main(void) {
    CHECK(test_sigmoid_bias_affects_selection_only() == 0);
    CHECK(test_softmax_is_stable_and_selects_top_k() == 0);
    CHECK(test_sqrt_softplus_scores() == 0);
    CHECK(test_ties_keep_lower_expert_index() == 0);
    CHECK(test_finalize_uses_effective_expert_count() == 0);
    puts("moe routing tests: ok");
    return 0;
}
