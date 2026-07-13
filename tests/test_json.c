#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../json.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    char error[128] = {0};
    jval *root = json_parse_full(
        "{\"name\":\"Colibri\\nCPU\",\"enabled\":true,\"empty\":null,"
        "\"values\":[1,-2.5,3e2],\"unicode\":\"\\u03bb \\uD83D\\uDE80\"}",
        error, sizeof(error)
    );

    CHECK(root && root->t == J_OBJ);
    CHECK(strcmp(json_get(root, "name")->str, "Colibri\nCPU") == 0);
    CHECK(json_get(root, "enabled")->boolean == 1);
    CHECK(json_get(root, "empty")->t == J_NULL);
    CHECK(json_get(root, "missing") == NULL);

    jval *values = json_get(root, "values");
    CHECK(values->t == J_ARR && values->len == 3);
    CHECK(values->kids[0]->num == 1.0);
    CHECK(values->kids[1]->num == -2.5);
    CHECK(values->kids[2]->num == 300.0);
    CHECK(strcmp(json_get(root, "unicode")->str, "λ 🚀") == 0);
    json_free(root);

    root = json_parse_full("{\"ok\":true} trailing", error, sizeof(error));
    CHECK(root == NULL);
    CHECK(strstr(error, "trailing") != NULL);

    root = json_parse_full("{\"broken\":", error, sizeof(error));
    CHECK(root == NULL);
    CHECK(error[0] != 0);

    puts("json tests: ok");
    return 0;
}
