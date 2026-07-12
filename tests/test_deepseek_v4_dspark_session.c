#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ds4.h"
#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <base.gguf> <dspark.gguf> <oracle-dir>\n", argv[0]);
        return 2;
    }

    shards oracle;
    st_init(&oracle, argv[3]);
    int64_t first_id = -1, input_id = -1, expected[6];
    st_read_raw(&oracle, "base.first_id", &first_id, 0);
    st_read_raw(&oracle, "base.input_id", &input_id, 0);
    st_read_raw(&oracle, "dspark.output_ids", expected, 0);

    char lock_path[] = "/tmp/floyd-dspark-session.XXXXXX";
    int lock_fd = mkstemp(lock_path);
    CHECK(lock_fd >= 0);
    CHECK(close(lock_fd) == 0);
    CHECK(setenv("DS4_LOCK_FILE", lock_path, 1) == 0);

    ds4_engine_options options = {
        .model_path = argv[1],
        .dspark_path = argv[2],
        .backend = DS4_BACKEND_METAL,
        .mtp_draft_tokens = 5,
    };
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    char error[4096] = {0};
    CHECK(ds4_engine_open(&engine, &options) == 0);
    CHECK(ds4_session_create(&session, engine, 64) == 0);

    int prompt_ids[] = {0, 128803, 33310, 128804, 128822};
    ds4_tokens prompt = {.v = prompt_ids, .len = 5, .cap = 5};
    CHECK(ds4_session_sync(session, &prompt, error, sizeof(error)) == 0);
    CHECK(ds4_session_argmax(session) == (int)first_id);
    CHECK(ds4_session_eval(session, (int)first_id, error, sizeof(error)) == 0);
    CHECK(ds4_session_argmax(session) == (int)input_id);

    int actual[6] = {-1, -1, -1, -1, -1, -1};
    CHECK(ds4_session_copy_dspark_proposals(session, actual, 6) == 6);
    int hits = 0;
    for (int i = 0; i < 6; i++) if (actual[i] == expected[i]) hits++;
    printf("DeepSeek V4 DSpark resident proposals: ids=%d/6\n", hits);
    CHECK(hits == 6);

    ds4_session_free(session);
    ds4_engine_close(engine);
    CHECK(unlink(lock_path) == 0);
    unsetenv("DS4_LOCK_FILE");
    return 0;
}
