#ifndef MOONLIGHT_ORACLE_H
#define MOONLIGHT_ORACLE_H

#include <stddef.h>

int moonlight_oracle_enabled(void);
int moonlight_oracle_write_f32(const char *name, const float *data, size_t count);

#endif
