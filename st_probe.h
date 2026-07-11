#ifndef ST_PROBE_H
#define ST_PROBE_H

#include <stdint.h>

#include "st.h"

enum { ST_DTYPE_COUNT = 7 };

static inline const char *st_dtype_name(int dtype) {
    switch (dtype) {
        case ST_DTYPE_BF16: return "BF16";
        case ST_DTYPE_F16: return "F16";
        case ST_DTYPE_F32: return "F32";
        case ST_DTYPE_U8: return "U8/I8";
        case ST_DTYPE_I64: return "I64";
        case ST_DTYPE_F8_E8M0: return "F8_E8M0";
        case ST_DTYPE_F8_E4M3: return "F8_E4M3";
        default: return "UNKNOWN";
    }
}

static inline void st_dtype_counts(const shards *source, uint64_t counts[ST_DTYPE_COUNT]) {
    for (int dtype = 0; dtype < ST_DTYPE_COUNT; dtype++) counts[dtype] = 0;
    for (int i = 0; i < source->n; i++) {
        int dtype = source->t[i].dtype;
        if (dtype >= 0 && dtype < ST_DTYPE_COUNT) counts[dtype]++;
    }
}

#endif
