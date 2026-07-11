/* Gate di parita' del tokenizer Moonlight: confronta mtok_encode/mtok_decode
 * contro l'oracolo tok_cases.json (generato da tools/make_tok_cases.py con
 * semantica tiktoken disallowed_special=()).
 * Uso: test_tok_moon <model_dir> <tok_cases.json> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../tok.h"        /* hmap, u8_next, tk_read_file */
#include "../tok_moon.h"

static void print_ids(const int *ids, int n) {
    printf("[");
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
    printf("]");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <tok_cases.json>\n", argv[0]);
        return 1;
    }

    MTok T;
    mtok_load(&T, argv[1]);

    long fn;
    char *buf = tk_read_file(argv[2], &fn);
    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    jval *cases = json_get(root, "cases");
    if (!cases || cases->t != J_ARR) {
        fprintf(stderr, "%s: manca l'array 'cases'\n", argv[2]);
        return 1;
    }

    const int MAXIDS = 4096;
    const int MAXOUT = 1 << 20;
    int *out = malloc(MAXIDS * sizeof(int));
    char *decbuf = malloc(MAXOUT);

    int ok = 0, tot = cases->len;
    for (int i = 0; i < cases->len; i++) {
        jval *c = cases->kids[i];
        jval *jtext = json_get(c, "text");
        jval *jids = json_get(c, "ids");
        if (!jtext || !jids || jids->t != J_ARR) {
            fprintf(stderr, "case %d: malformed (missing text/ids)\n", i);
            continue;
        }
        const char *txt = jtext->str;
        int tlen = (int)strlen(txt);
        int n_exp = jids->len;
        int *exp = malloc((n_exp > 0 ? n_exp : 1) * sizeof(int));
        for (int k = 0; k < n_exp; k++) exp[k] = (int)jids->kids[k]->num;

        int n_act = mtok_encode(&T, txt, tlen, out, MAXIDS);
        int enc_ok = (n_act == n_exp);
        if (enc_ok) {
            for (int k = 0; k < n_exp; k++) {
                if (out[k] != exp[k]) { enc_ok = 0; break; }
            }
        }

        int declen = mtok_decode(&T, exp, n_exp, decbuf, MAXOUT);
        int dec_ok = (declen == tlen) && (tlen == 0 || memcmp(decbuf, txt, tlen) == 0);

        if (enc_ok && dec_ok) {
            ok++;
        } else {
            printf("FAIL case %d: text=\"%s\"\n", i, txt);
            printf("  expected ids: "); print_ids(exp, n_exp); printf("\n");
            printf("  actual ids:   "); print_ids(out, n_act); printf("\n");
            if (!dec_ok) {
                printf("  decode(expected) len=%d (want %d): \"%.*s\"\n", declen, tlen, declen, decbuf);
            }
        }
        free(exp);
    }
    printf("tokenizer parity: %d/%d\n", ok, tot);

    /* Template (Chat-Task 4 fa la costruzione completa): qui solo decode-direction
     * sui 42 id del template, verificando che il testo renderizzato contenga i
     * marcatori di ruolo e il contenuto del primo turno utente. */
    int tmpl_pass = 1;
    jval *tmpl = json_get(root, "template");
    if (tmpl) {
        jval *tids = json_get(tmpl, "ids");
        if (tids && tids->t == J_ARR) {
            int tn = tids->len;
            int *tarr = malloc((tn > 0 ? tn : 1) * sizeof(int));
            for (int k = 0; k < tn; k++) tarr[k] = (int)tids->kids[k]->num;
            int dl = mtok_decode(&T, tarr, tn, decbuf, MAXOUT);
            if (dl < 0 || dl >= MAXOUT) dl = MAXOUT - 1;
            decbuf[dl] = 0;
            if (!strstr(decbuf, "<|im_system|>") || !strstr(decbuf, "What is 2+2?")) {
                tmpl_pass = 0;
                printf("FAIL template roundtrip: decoded=\"%s\"\n", decbuf);
            }
            free(tarr);
        } else {
            tmpl_pass = 0;
        }
    }
    printf("template roundtrip: %s\n", tmpl_pass ? "ok" : "FAIL");

    int all_ok = (ok == tot) && tmpl_pass;
    if (!all_ok) return 1;
    return 0;
}
