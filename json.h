/* Parser JSON minimale, header-only. Serve per:
 *  - l'header dei file safetensors (un grande oggetto nome->{dtype,shape,data_offsets})
 *  - ref.json (per leggere prompt_ids / full_ids)
 * Non e' completo (niente unicode \uXXXX, niente notazione esotica) ma copre cio' che serve. */
#ifndef JSON_H
#define JSON_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdint.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype t;
    double num;            /* J_NUM */
    int    boolean;        /* J_BOOL */
    char  *str;            /* J_STR (NUL-terminata, dentro l'arena) */
    /* array: figli in [0..len); oggetto: chiavi[] e figli[] in parallelo */
    struct jval **kids;
    char        **keys;    /* solo per J_OBJ */
    int           len;
} jval;

typedef struct {
    const char *s;
    char       *arena;     /* buffer per le stringhe smontate */
    size_t      acap, aoff;
    const char *error;
} jparser;

static char *j_dup(jparser *p, const char *b, int n) {
    /* ogni stringa ha la sua allocazione: un'arena con realloc sposterebbe il
     * buffer invalidando i puntatori gia' emessi (use-after-free). */
    (void)p;
    char *d = (char *)malloc(n + 1);
    memcpy(d, b, n); d[n] = 0;
    return d;
}

static void j_ws(jparser *p) { while (*p->s && isspace((unsigned char)*p->s)) p->s++; }

static jval *j_new(jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    v->t = t; return v;
}

static jval *j_parse_val(jparser *p);

static void json_free(jval *v);

static void j_fail(jparser *p, const char *message) {
    if (!p->error) p->error = message;
}

static int j_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int j_grow(char **buf, size_t *cap, size_t need) {
    if (need <= *cap) return 1;
    size_t next = *cap ? *cap : 64;
    while (next < need) {
        if (next > SIZE_MAX / 2) return 0;
        next *= 2;
    }
    char *grown = (char *)realloc(*buf, next);
    if (!grown) return 0;
    *buf = grown;
    *cap = next;
    return 1;
}

static int j_put(char **buf, size_t *len, size_t *cap, unsigned char c) {
    if (!j_grow(buf, cap, *len + 2)) return 0;
    (*buf)[(*len)++] = (char)c;
    return 1;
}

static int j_put_utf8(char **buf, size_t *len, size_t *cap, unsigned cp) {
    if (cp < 0x80) return j_put(buf, len, cap, (unsigned char)cp);
    if (cp < 0x800)
        return j_put(buf, len, cap, 0xC0 | (cp >> 6)) &&
               j_put(buf, len, cap, 0x80 | (cp & 0x3F));
    if (cp < 0x10000)
        return j_put(buf, len, cap, 0xE0 | (cp >> 12)) &&
               j_put(buf, len, cap, 0x80 | ((cp >> 6) & 0x3F)) &&
               j_put(buf, len, cap, 0x80 | (cp & 0x3F));
    return j_put(buf, len, cap, 0xF0 | (cp >> 18)) &&
           j_put(buf, len, cap, 0x80 | ((cp >> 12) & 0x3F)) &&
           j_put(buf, len, cap, 0x80 | ((cp >> 6) & 0x3F)) &&
           j_put(buf, len, cap, 0x80 | (cp & 0x3F));
}

static char *j_parse_str_raw(jparser *p) {
    if (*p->s != '"') {
        j_fail(p, "expected string");
        return NULL;
    }
    p->s++;
    char *out = NULL;
    size_t len = 0, cap = 0;
    while (*p->s && *p->s != '"') {
        unsigned char c = (unsigned char)*p->s++;
        if (c < 0x20) {
            j_fail(p, "unescaped control character in string");
            free(out);
            return NULL;
        }
        if (c == '\\' && *p->s) {
            char e = *p->s++;
            switch (e) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break;
                case 'r': c = '\r'; break; case 'b': c = '\b'; break;
                case 'f': c = '\f'; break; case '/': c = '/'; break;
                case '\\': c = '\\'; break; case '"': c = '"'; break;
                case 'u': {
                    unsigned cp = 0;
                    for (int i = 0; i < 4; i++) {
                        if (!p->s[i]) {
                            j_fail(p, "incomplete unicode escape");
                            free(out);
                            return NULL;
                        }
                        int h = j_hex(p->s[i]);
                        if (h < 0) {
                            j_fail(p, "invalid unicode escape");
                            free(out);
                            return NULL;
                        }
                        cp = (cp << 4) | (unsigned)h;
                    }
                    p->s += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (p->s[0] != '\\' || p->s[1] != 'u') {
                            j_fail(p, "missing low unicode surrogate");
                            free(out);
                            return NULL;
                        }
                        unsigned lo = 0;
                        for (int i = 0; i < 4; i++) {
                            if (!p->s[i + 2]) {
                                j_fail(p, "incomplete low unicode surrogate");
                                free(out);
                                return NULL;
                            }
                            int h = j_hex(p->s[i + 2]);
                            if (h < 0) {
                                j_fail(p, "invalid low unicode surrogate");
                                free(out);
                                return NULL;
                            }
                            lo = (lo << 4) | (unsigned)h;
                        }
                        if (lo < 0xDC00 || lo > 0xDFFF) {
                            j_fail(p, "invalid low unicode surrogate");
                            free(out);
                            return NULL;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        p->s += 6;
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        j_fail(p, "unexpected low unicode surrogate");
                        free(out);
                        return NULL;
                    }
                    if (!j_put_utf8(&out, &len, &cap, cp)) {
                        j_fail(p, "out of memory parsing string");
                        free(out);
                        return NULL;
                    }
                    continue;
                }
                default:
                    j_fail(p, "invalid string escape");
                    free(out);
                    return NULL;
            }
        } else if (c == '\\') {
            j_fail(p, "unterminated string escape");
            free(out);
            return NULL;
        }
        if (!j_put(&out, &len, &cap, c)) {
            j_fail(p, "out of memory parsing string");
            free(out);
            return NULL;
        }
    }
    if (*p->s != '"') {
        j_fail(p, "unterminated string");
        free(out);
        return NULL;
    }
    p->s++;
    if (!j_grow(&out, &cap, len + 1)) {
        j_fail(p, "out of memory parsing string");
        free(out);
        return NULL;
    }
    out[len] = 0;
    return out;
}

static jval *j_parse_val(jparser *p) {
    j_ws(p);
    if (p->error) return NULL;
    char c = *p->s;
    if (c == '"') { jval *v = j_new(J_STR); v->str = j_parse_str_raw(p); return v; }
    if (c == '{') {
        p->s++; jval *v = j_new(J_OBJ);
        int cap = 8; v->keys = malloc(cap * sizeof(char*)); v->kids = malloc(cap * sizeof(jval*));
        j_ws(p);
        if (*p->s == '}') { p->s++; return v; }
        for (;;) {
            j_ws(p);
            if (*p->s != '"') { j_fail(p, "expected object key"); return v; }
            char *key = j_parse_str_raw(p);
            if (p->error) { free(key); return v; }
            j_ws(p);
            if (*p->s != ':') { free(key); j_fail(p, "expected colon"); return v; }
            p->s++;
            jval *val = j_parse_val(p);
            if (p->error || !val) { free(key); json_free(val); return v; }
            if (v->len == cap) { cap *= 2; v->keys = realloc(v->keys, cap*sizeof(char*)); v->kids = realloc(v->kids, cap*sizeof(jval*)); }
            v->keys[v->len] = key; v->kids[v->len] = val; v->len++;
            j_ws(p);
            if (*p->s == ',') { p->s++; continue; }
            if (*p->s == '}') { p->s++; break; }
            j_fail(p, "expected comma or object end");
            break;
        }
        return v;
    }
    if (c == '[') {
        p->s++; jval *v = j_new(J_ARR);
        int cap = 8; v->kids = malloc(cap * sizeof(jval*));
        j_ws(p);
        if (*p->s == ']') { p->s++; return v; }
        for (;;) {
            jval *val = j_parse_val(p);
            if (p->error || !val) { json_free(val); return v; }
            if (v->len == cap) { cap *= 2; v->kids = realloc(v->kids, cap*sizeof(jval*)); }
            v->kids[v->len++] = val;
            j_ws(p);
            if (*p->s == ',') { p->s++; continue; }
            if (*p->s == ']') { p->s++; break; }
            j_fail(p, "expected comma or array end");
            break;
        }
        return v;
    }
    if (!strncmp(p->s, "true", 4)) { p->s += 4; jval *v = j_new(J_BOOL); v->boolean = 1; return v; }
    if (!strncmp(p->s, "false", 5)) { p->s += 5; jval *v = j_new(J_BOOL); v->boolean = 0; return v; }
    if (!strncmp(p->s, "null", 4)) { p->s += 4; return j_new(J_NULL); }
    /* numero */
    { char *end; double d = strtod(p->s, &end);
      if (end == p->s) { j_fail(p, "expected JSON value"); return NULL; }
      p->s = end; jval *v = j_new(J_NUM); v->num = d; return v; }
}

/* API */
static void json_free(jval *v) {
    if (!v) return;
    if (v->t == J_STR) free(v->str);
    if (v->t == J_OBJ) {
        for (int i = 0; i < v->len; i++) free(v->keys[i]);
    }
    if (v->t == J_OBJ || v->t == J_ARR) {
        for (int i = 0; i < v->len; i++) json_free(v->kids[i]);
        free(v->kids);
    }
    free(v->keys);
    free(v);
}

static jval *json_parse_full(const char *text, char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!text) {
        if (error && error_size) snprintf(error, error_size, "null JSON input");
        return NULL;
    }
    jparser p = { text, NULL, 0, 0, NULL };
    jval *v = j_parse_val(&p);
    j_ws(&p);
    if (!p.error && *p.s) p.error = "trailing data after JSON value";
    if (p.error || !v) {
        if (error && error_size)
            snprintf(error, error_size, "%s", p.error ? p.error : "invalid JSON");
        json_free(v);
        return NULL;
    }
    return v;
}

static jval *json_parse(const char *text, char **arena_out) {
    if (arena_out) *arena_out = NULL;
    return json_parse_full(text, NULL, 0);
}

static jval *json_get(jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (strcmp(o->keys[i], key) == 0) return o->kids[i];
    return NULL;
}

#endif
