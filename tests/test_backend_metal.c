#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../backend_metal.h"

static float frand(void){ return (float)rand()/RAND_MAX*2.f-1.f; }

static double q8_maxrel(const float *y, const int8_t *w8, const float *s, const float *x, int O, int I, int S){
    double maxrel=0;
    for(int t=0;t<S;t++) for(int o=0;o<O;o++){
        double ref=0; for(int i=0;i<I;i++) ref+=(double)w8[(size_t)o*I+i]*x[(size_t)t*I+i];
        ref*=s[o];
        double d=fabs(y[(size_t)t*O+o]-ref), rel=d/(fabs(ref)+1e-6);
        if(rel>maxrel) maxrel=rel;
    }
    return maxrel;
}

static double q4_maxrel(const float *y, const uint8_t *w4, const float *s, const float *x, int O, int I, int S){
    int Ib=(I+1)/2;
    double maxrel=0;
    for(int t=0;t<S;t++) for(int o=0;o<O;o++){
        double ref=0;
        for(int i=0;i+1<I;i+=2){ uint8_t b=w4[(size_t)o*Ib+(i>>1)];
            ref+=((int)(b&0xF)-8)*(double)x[(size_t)t*I+i]+((int)(b>>4)-8)*(double)x[(size_t)t*I+i+1]; }
        if(I&1){ uint8_t b=w4[(size_t)o*Ib+(I>>1)]; ref+=((int)(b&0xF)-8)*(double)x[(size_t)t*I+I-1]; }
        ref*=s[o];
        double d=fabs(y[(size_t)t*O+o]-ref), rel=d/(fabs(ref)+1e-6);
        if(rel>maxrel) maxrel=rel;
    }
    return maxrel;
}

int main(void){
    if(!fm_init()){ fprintf(stderr,"metal non disponibile\n"); return 1; }
    printf("device: %s\n", fm_device_name());
    srand(42);
    int O=128, I=257, S=4;                      /* I dispari: esercita il tail q4 */

    /* --- q8, cache=0: buffer pesi transiente (percorso "slab expert") --- */
    int8_t *w8=malloc((size_t)O*I); float *s=malloc(O*4), *x=malloc((size_t)S*I*4);
    for(int i=0;i<O*I;i++) w8[i]=rand()%255-127;
    for(int o=0;o<O;o++) s[o]=frand()*0.01f+0.02f;
    for(int i=0;i<S*I;i++) x[i]=frand();
    float *y=malloc((size_t)S*O*4);
    fm_matmul_q8(y,x,w8,s,O,I,S,0);
    double maxrel=q8_maxrel(y,w8,s,x,O,I,S);
    printf("q8 max rel err (cache=0): %.2e\n", maxrel);
    if(maxrel>1e-3){ printf("FAIL q8 cache=0\n"); return 1; }

    /* stesso identico pointer w8, di nuovo cache=0: deve restare corretto e
     * NON deve inserirsi nella cache (percorso transiente puro). */
    for(int i=0;i<S*I;i++) x[i]=frand();
    fm_matmul_q8(y,x,w8,s,O,I,S,0);
    maxrel=q8_maxrel(y,w8,s,x,O,I,S);
    printf("q8 max rel err (cache=0, ripetuto): %.2e\n", maxrel);
    if(maxrel>1e-3){ printf("FAIL q8 cache=0 ripetuto\n"); return 1; }

    /* --- q4, cache=1: buffer pesi cache-ato per puntatore (percorso "tensore
     * model-resident") --- */
    int Ib=(I+1)/2; uint8_t *w4=malloc((size_t)O*Ib);
    for(int i=0;i<O*Ib;i++) w4[i]=rand()&0xFF;
    fm_matmul_q4(y,x,w4,s,O,I,S,1);
    maxrel=q4_maxrel(y,w4,s,x,O,I,S);
    printf("q4 max rel err (cache=1): %.2e\n", maxrel);
    if(maxrel>1e-3){ printf("FAIL q4 cache=1\n"); return 1; }

    /* Chiamata ripetuta con lo STESSO puntatore w4 (e la stessa s), cache=1:
     * deve pescare il buffer dalla cache interna (chiave-puntatore) invece
     * di ricopiare — verifica che il riuso non rompa nulla e che x (sempre
     * transiente) non resti "incollato" al valore precedente. */
    for(int i=0;i<S*I;i++) x[i]=frand();
    fm_matmul_q4(y,x,w4,s,O,I,S,1);
    maxrel=q4_maxrel(y,w4,s,x,O,I,S);
    printf("q4 max rel err (cache=1, ripetuto stesso puntatore): %.2e\n", maxrel);
    if(maxrel>1e-3){ printf("FAIL q4 cache=1 ripetuto\n"); return 1; }

    printf("OK\n"); return 0;
}
