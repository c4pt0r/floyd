#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../tok.h"
#include "../v4_chat_format.h"

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model-dir> <oracle.json>\n", argv[0]);
        return 2;
    }
    char tokenizer_path[2048];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", argv[1]);
    Tok tokenizer;
    tok_load(&tokenizer, tokenizer_path);
    CHECK(tok_id_of(&tokenizer, "<｜begin▁of▁sentence｜>") == 0);
    CHECK(tok_id_of(&tokenizer, "<｜end▁of▁sentence｜>") == 1);

    long input_size;
    char *input = tk_read_file(argv[2], &input_size);
    char *arena = NULL;
    jval *root = json_parse(input, &arena);
    jval *cases = json_get(root, "cases");
    CHECK(cases && cases->t == J_ARR);

    char rendered[4096];
    size_t rendered_size = v4_chat_append_user(
        rendered, sizeof(rendered), 0, "hello", 1);
    CHECK(rendered_size != SIZE_MAX);
    CHECK(rendered_size == strlen(json_get(cases->kids[0], "prompt")->str));
    CHECK(!memcmp(rendered, json_get(cases->kids[0], "prompt")->str,
                  rendered_size));

    int actual[4096];
    char decoded[1 << 16];
    for (int c = 0; c < cases->len; c++) {
        jval *item = cases->kids[c];
        jval *prompt = json_get(item, "prompt");
        jval *ids = json_get(item, "ids");
        jval *expected_decoded = json_get(item, "decoded");
        CHECK(prompt && ids && expected_decoded && ids->t == J_ARR);
        int count = tok_encode(&tokenizer, prompt->str,
                               (int)strlen(prompt->str), actual, 4096);
        if (count != ids->len) {
            fprintf(stderr, "case %d token count: got %d want %d\n",
                    c, count, ids->len);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            int expected = (int)ids->kids[i]->num;
            if (actual[i] != expected) {
                fprintf(stderr,
                        "case %d token %d: got %d want %d\n",
                        c, i, actual[i], expected);
                return 1;
            }
        }
        int decoded_size = tok_decode(&tokenizer, actual, count,
                                      decoded, sizeof(decoded) - 1);
        decoded[decoded_size] = 0;
        if (strcmp(decoded, expected_decoded->str)) {
            fprintf(stderr, "case %d decode mismatch\n", c);
            return 1;
        }
    }
    printf("v4 chat tokenizer parity: %d/%d\n", cases->len, cases->len);
    return 0;
}
