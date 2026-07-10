/* backend_metal.h — interfaccia C-callable del backend Metal (compute-only kernel).
 * Implementazione in backend_metal.m (Objective-C, compilato separatamente col
 * proprio target nel Makefile). Nessuna dipendenza qui su Metal/Foundation:
 * questo header e' incluso anche da floyd.c (Task 7) e dal test in tests/. */
#ifndef BACKEND_METAL_H
#define BACKEND_METAL_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 = device+kernel pronti (compilazione MSL ok); 0 = non disponibile
 * (il chiamante decide se fare fallback CPU o uscire con errore). */
int  fm_init(void);
const char *fm_device_name(void);

/* Semantica identica a matmul_qt fmt1 (q8) / fmt2 (q4) lato CPU:
 * y[S,O] = x[S,I] @ w[O,I]^T, scala per riga s[O]; int4: nibble basso
 * per primo in ogni byte, offset -8.
 *
 * cache: 1 = il buffer pesi (w, e la scala s) e' cercato/inserito nella cache
 *        interna MTLBuffer chiave-puntatore (per tensori residenti stabili:
 *        upload lazy una sola volta, poi riuso — vedi colibri' CUDA "lazy
 *        upload once"). 0 = buffer transiente creato per questa sola chiamata
 *        (newBufferWithBytes, rilasciato con l'autoreleasepool) — per le
 *        "slab" di expert le cui zone host vengono riciclate dalla LRU: un
 *        buffer cache-ato su un puntatore riciclato servirebbe pesi stantii
 *        senza errore visibile. */
void fm_matmul_q8(float *y, const float *x, const int8_t  *w, const float *s,
                   int O, int I, int S, int cache);
void fm_matmul_q4(float *y, const float *x, const uint8_t *w, const float *s,
                   int O, int I, int S, int cache);

#ifdef __cplusplus
}
#endif
#endif
