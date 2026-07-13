#include "moonlight_oracle.h"

#include <ctype.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *oracle_directory(void) {
    const char *directory = getenv("FLOYD_MOONLIGHT_ORACLE_DIR");
    return directory && directory[0] ? directory : NULL;
}

static int valid_name(const char *name) {
    if (!name || !name[0]) return 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        if (!isalnum(*p) && *p != '.' && *p != '_' && *p != '-') return 0;
    }
    return strstr(name, "..") == NULL;
}

static int write_all(int file, const void *data, size_t size) {
    const uint8_t *cursor = data;
    while (size) {
        ssize_t written = write(file, cursor, size);
        if (written <= 0) return 0;
        cursor += written;
        size -= (size_t)written;
    }
    return 1;
}

int moonlight_oracle_enabled(void) {
    return oracle_directory() != NULL;
}

int moonlight_oracle_write_f32(const char *name, const float *data, size_t count) {
    const char *directory = oracle_directory();
    char path[4096];
    char temporary[4096];
    int file;
    int ok;

    if (!directory) return 1;
    if (!valid_name(name) || (!data && count)) return 0;
    if (count > SIZE_MAX / sizeof(float)) return 0;
    if (snprintf(path, sizeof(path), "%s/%s.f32", directory, name) >=
        (int)sizeof(path)) return 0;
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.%ld", path,
                 (long)getpid()) >= (int)sizeof(temporary)) return 0;

    file = open(temporary, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (file < 0) return 0;
    ok = write_all(file, data, count * sizeof(float));
    if (ok) ok = fsync(file) == 0;
    if (close(file) != 0) ok = 0;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) unlink(temporary);
    return ok;
}
