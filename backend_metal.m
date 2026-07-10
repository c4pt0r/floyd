/* backend_metal.m — wrapper Objective-C sopra Metal.framework per i due kernel
 * di matmul quantizzato (q8/q4) definiti in kernels.metal. Nessun collegamento
 * a floyd.c qui (integrazione motore: Task 7); solo API C-callable + test
 * a livello kernel contro il riferimento CPU (tests/test_backend_metal.c). */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend_metal.h"
#include "kernels_metal.h"      /* xxd -i kernels.metal: unsigned char kernels_metal[], kernels_metal_len */

static id<MTLDevice> g_dev; static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_p8, g_p4;

/* Cache pesi chiave-puntatore: per tensori model-resident, upload lazy una
 * sola volta poi riuso (equivalente a colibri' CUDA "lazy upload once").
 * NON va usata per viste su slab expert la cui memoria host viene riciclata
 * dalla LRU — un buffer cache-ato su un puntatore riciclato servirebbe pesi
 * stantii senza errore visibile; per quei casi il chiamante passa cache=0. */
#define FM_WCACHE 4096
static struct { const void *host; id<MTLBuffer> buf; } g_wc[FM_WCACHE];
static int g_nwc;

int fm_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;
        g_q = [g_dev newCommandQueue];
        NSString *src = [[NSString alloc] initWithBytes:kernels_metal length:kernels_metal_len encoding:NSUTF8StringEncoding];
        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "[METAL] compile: %s\n", err.localizedDescription.UTF8String); return 0; }
        g_p8 = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matmul_q8"] error:&err];
        g_p4 = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matmul_q4"] error:&err];
        return (g_p8 && g_p4) ? 1 : 0;
    }
}
const char *fm_device_name(void) { return g_dev ? g_dev.name.UTF8String : "(none)"; }

/* Lookup/insert nella cache chiave-puntatore. */
static id<MTLBuffer> wbuf_cached(const void *host, size_t len) {
    for (int i = 0; i < g_nwc; i++) if (g_wc[i].host == host) return g_wc[i].buf;
    id<MTLBuffer> b = [g_dev newBufferWithBytes:host length:len options:MTLResourceStorageModeShared];
    if (g_nwc < FM_WCACHE) { g_wc[g_nwc].host = host; g_wc[g_nwc].buf = b; g_nwc++; }
    return b;
}

/* cache=1 -> buffer stabile riusabile (vedi wbuf_cached); cache=0 -> buffer
 * transiente per questa sola chiamata, mai inserito in cache. */
static id<MTLBuffer> wbuf(const void *host, size_t len, int cache) {
    if (cache) return wbuf_cached(host, len);
    return [g_dev newBufferWithBytes:host length:len options:MTLResourceStorageModeShared];
}

static void run(id<MTLComputePipelineState> p, const void *w, size_t wlen,
                const float *s, const float *x, float *y, int O, int I, int S, int cache) {
    @autoreleasepool {
        id<MTLBuffer> bw = wbuf(w, wlen, cache);
        id<MTLBuffer> bs = wbuf(s, (size_t)O * 4, cache);
        id<MTLBuffer> bx = [g_dev newBufferWithBytes:x length:(size_t)S * I * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_dev newBufferWithLength:(size_t)S * O * 4 options:MTLResourceStorageModeShared];
        uint32_t d[3] = { (uint32_t)O, (uint32_t)I, (uint32_t)S };
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1];
        [e setBuffer:bx offset:0 atIndex:2]; [e setBuffer:by offset:0 atIndex:3];
        [e setBytes:d length:sizeof(d) atIndex:4];
        MTLSize grid = MTLSizeMake(O, S, 1);
        NSUInteger tw = p.maxTotalThreadsPerThreadgroup > 256 ? 256 : p.maxTotalThreadsPerThreadgroup;
        [e dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(y, by.contents, (size_t)S * O * 4);
    }
}
void fm_matmul_q8(float *y, const float *x, const int8_t *w, const float *s, int O, int I, int S, int cache)
{ run(g_p8, w, (size_t)O * I, s, x, y, O, I, S, cache); }
void fm_matmul_q4(float *y, const float *x, const uint8_t *w, const float *s, int O, int I, int S, int cache)
{ run(g_p4, w, (size_t)O * ((I + 1) / 2), s, x, y, O, I, S, cache); }
