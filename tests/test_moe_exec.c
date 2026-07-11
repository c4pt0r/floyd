#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../moe_exec.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define CHECK_NEAR(actual, expected, tolerance) \
    CHECK(fabsf((actual) - (expected)) <= (tolerance))

static float test_silu(float x) {
    return x / (1.0f + expf(-x));
}

static int test_learned_route_and_shared_expert(void) {
    const float router[] = {0.0f, 1.0f};
    const float e0_w1[] = {1.0f}, e0_w2[] = {1.0f}, e0_w3[] = {1.0f};
    const float e1_w1[] = {0.5f}, e1_w2[] = {2.0f}, e1_w3[] = {1.0f};
    const float sh_w1[] = {0.5f}, sh_w2[] = {1.0f}, sh_w3[] = {0.5f};
    const MoeExpertF32 experts[] = {
        {.w1 = e0_w1, .w2 = e0_w2, .w3 = e0_w3},
        {.w1 = e1_w1, .w2 = e1_w2, .w3 = e1_w3},
    };
    const MoeExpertF32 shared[] = {
        {.w1 = sh_w1, .w2 = sh_w2, .w3 = sh_w3},
    };
    const MoeF32 model = {
        .hidden = 1, .intermediate = 1, .n_experts = 2, .top_k = 1,
        .score_fn = MOE_SCORE_SQRT_SOFTPLUS, .normalize = 1,
        .route_scale = 1.0f, .swiglu_limit = 10.0f,
        .router = router, .experts = experts,
        .shared_experts = shared, .n_shared_experts = 1,
    };
    const float input[] = {2.0f};
    const int token_ids[] = {0};
    float output[1];

    CHECK(moe_f32_forward(&model, input, token_ids, 1, output));
    CHECK_NEAR(output[0], 5.0f * test_silu(1.0f), 1e-6f);
    return 0;
}

static int test_hash_route_uses_token_table(void) {
    const float router[] = {0.0f, 1.0f};
    const int64_t fixed[] = {0, 1};
    const float e0_w1[] = {1.0f}, e0_w2[] = {1.0f}, e0_w3[] = {1.0f};
    const float e1_w1[] = {0.5f}, e1_w2[] = {2.0f}, e1_w3[] = {1.0f};
    const MoeExpertF32 experts[] = {
        {.w1 = e0_w1, .w2 = e0_w2, .w3 = e0_w3},
        {.w1 = e1_w1, .w2 = e1_w2, .w3 = e1_w3},
    };
    const MoeF32 model = {
        .hidden = 1, .intermediate = 1, .n_experts = 2, .top_k = 1,
        .vocab_size = 2, .score_fn = MOE_SCORE_SQRT_SOFTPLUS,
        .normalize = 1, .route_scale = 1.0f, .swiglu_limit = 10.0f,
        .router = router, .fixed_experts = fixed, .experts = experts,
    };
    const float input[] = {2.0f};
    const int token_ids[] = {0};
    float output[1];

    CHECK(moe_f32_forward(&model, input, token_ids, 1, output));
    CHECK_NEAR(output[0], 2.0f * test_silu(2.0f), 1e-6f);
    return 0;
}

static int test_clamped_swiglu(void) {
    const float router[] = {0.0f};
    const float w1[] = {2.0f}, w2[] = {1.0f}, w3[] = {-2.0f};
    const MoeExpertF32 experts[] = {{.w1 = w1, .w2 = w2, .w3 = w3}};
    const MoeF32 model = {
        .hidden = 1, .intermediate = 1, .n_experts = 1, .top_k = 1,
        .score_fn = MOE_SCORE_SOFTMAX, .normalize = 1,
        .route_scale = 1.0f, .swiglu_limit = 10.0f,
        .router = router, .experts = experts,
    };
    const float input[] = {10.0f};
    const int token_ids[] = {0};
    float output[1];

    CHECK(moe_f32_forward(&model, input, token_ids, 1, output));
    CHECK_NEAR(output[0], -10.0f * test_silu(10.0f), 1e-4f);
    return 0;
}

static int test_rejects_missing_shared_experts(void) {
    const float router[] = {0.0f};
    const float weight[] = {1.0f};
    const MoeExpertF32 experts[] = {{.w1 = weight, .w2 = weight, .w3 = weight}};
    const MoeF32 model = {
        .hidden = 1, .intermediate = 1, .n_experts = 1, .top_k = 1,
        .score_fn = MOE_SCORE_SOFTMAX, .normalize = 1, .route_scale = 1.0f,
        .router = router, .experts = experts,
        .shared_experts = NULL, .n_shared_experts = 1,
    };
    const float input[] = {1.0f};
    float output[1];

    CHECK(!moe_f32_forward(&model, input, NULL, 1, output));
    return 0;
}

int main(void) {
    CHECK(test_learned_route_and_shared_expert() == 0);
    CHECK(test_hash_route_uses_token_table() == 0);
    CHECK(test_clamped_swiglu() == 0);
    CHECK(test_rejects_missing_shared_experts() == 0);
    puts("moe execution tests: ok");
    return 0;
}
