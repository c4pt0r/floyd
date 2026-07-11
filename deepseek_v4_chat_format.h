#ifndef DEEPSEEK_V4_CHAT_FORMAT_H
#define DEEPSEEK_V4_CHAT_FORMAT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline size_t deepseek_v4_chat_append_bytes(char *dst, size_t cap, size_t used,
                                           const char *source, size_t count) {
    if (!dst || !source || used >= cap || count >= cap - used) return SIZE_MAX;
    memcpy(dst + used, source, count);
    used += count;
    dst[used] = 0;
    return used;
}

static inline size_t deepseek_v4_chat_append_user(char *dst, size_t cap, size_t used,
                                          const char *text, int first_turn) {
    static const char bos[] = "<｜begin▁of▁sentence｜>";
    static const char user[] = "<｜User｜>";
    static const char assistant[] = "<｜Assistant｜></think>";
    if (!text) return SIZE_MAX;
    if (first_turn) {
        used = deepseek_v4_chat_append_bytes(dst, cap, used, bos, sizeof(bos) - 1);
        if (used == SIZE_MAX) return SIZE_MAX;
    }
    used = deepseek_v4_chat_append_bytes(dst, cap, used, user, sizeof(user) - 1);
    if (used == SIZE_MAX) return SIZE_MAX;
    used = deepseek_v4_chat_append_bytes(dst, cap, used, text, strlen(text));
    if (used == SIZE_MAX) return SIZE_MAX;
    return deepseek_v4_chat_append_bytes(dst, cap, used, assistant,
                                sizeof(assistant) - 1);
}

#endif
