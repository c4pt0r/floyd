/* Tokenizer Moonlight in C puro: tiktoken (BPE su byte grezzi, rank = ordine
 * del vocabolario) + pre-tokenizer "moonshot" (variante di cl100k con
 * alternative maiuscolo/minuscolo separate ed esclusione di Han).
 * Replica fedele di:
 *   - tiktoken.model: righe "base64(bytes) rank" (rank = id, 0..n_base-1)
 *   - tokenizer_config.json: added_tokens_decoder (id stringa -> {content})
 *   - pat_str (tokenization_moonshot.py):
 *       1) [\p{Han}]+
 *       2) [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&non-Han]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&non-Han]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *       3) [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&non-Han]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&non-Han]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *       4) \p{N}{1,3}
 *       5)  ?[^\s\p{L}\p{N}]+[\r\n]*
 *       6) \s*[\r\n]+   7) \s+(?!\S)   8) \s+
 * mtok_encode NON fa scansione degli special token (semantica
 * disallowed_special=() di tiktoken: gli special sono iniettati solo dal
 * template, mai riconosciuti dentro testo libero).
 * Richiede tok.h incluso PRIMA (hmap/hm_init/hm_put/hm_get, u8_next,
 * tk_read_file) e tok_unicode.h (is_L/is_N/is_S/is_Han/is_LuLt/is_Ll/is_Lm/is_Lo/is_M).
 */
#ifndef TOK_MOON_H
#define TOK_MOON_H
#ifndef TOK_H
#error "tok_moon.h va incluso DOPO tok.h"
#endif

typedef struct { unsigned char *b; int len; } MEnt;
typedef struct { char *str; int len; int id; } MSpecial;
typedef struct {
    hmap rank;              /* sequenza di byte -> id (0..n_base-1, rank == id) */
    MEnt *id2b; int n_base; /* id -> sequenza di byte (decode) */
    MSpecial *sp; int nsp;  /* added_tokens_decoder, ordinati per id crescente */
    int n_ids;              /* n_base + nsp */
} MTok;

/* ---------- base64 (alfabeto standard, senza dipendenze esterne) ---------- */
static int m_b64val(unsigned char c){
    if(c>='A'&&c<='Z') return c-'A';
    if(c>='a'&&c<='z') return c-'a'+26;
    if(c>='0'&&c<='9') return c-'0'+52;
    if(c=='+') return 62;
    if(c=='/') return 63;
    return -1;
}
static int m_b64_decode(const char *s, unsigned char **out, int *outlen){
    int n=(int)strlen(s);
    unsigned char *buf=(unsigned char*)malloc(n>0?n:1);
    int bo=0; uint32_t acc=0; int bits=0;
    for(int i=0;i<n;i++){
        char c=s[i];
        if(c=='=') break;
        int v=m_b64val((unsigned char)c);
        if(v<0){ free(buf); return 0; }
        acc=(acc<<6)|(uint32_t)v; bits+=6;
        if(bits>=8){ bits-=8; buf[bo++]=(unsigned char)((acc>>bits)&0xFF); }
    }
    *out=buf; *outlen=bo;
    return 1;
}

/* ---------- caricamento ---------- */
static int m_cmp_sp_id(const void *a, const void *b){
    return ((const MSpecial*)a)->id - ((const MSpecial*)b)->id;
}

static void mtok_load(MTok *T, const char *model_dir){
    memset(T,0,sizeof(*T));
    char path[4096];
    snprintf(path,sizeof(path),"%s/tiktoken.model",model_dir);
    FILE *f=fopen(path,"rb");
    if(!f){ perror(path); exit(1); }

    int cap=0, n=0; MEnt *ents=NULL;
    char linebuf[8192];
    int lineno=0;
    while(fgets(linebuf,sizeof(linebuf),f)){
        lineno++;
        int L=(int)strlen(linebuf);
        while(L>0 && (linebuf[L-1]=='\n'||linebuf[L-1]=='\r')) linebuf[--L]=0;
        if(L==0) continue;
        char *sep=strchr(linebuf,' ');
        if(!sep){ fprintf(stderr,"%s:%d: riga senza separatore ' ' tra base64 e rank\n",path,lineno); exit(1); }
        *sep=0;
        const char *b64=linebuf; const char *rankstr=sep+1;
        char *end; long rank=strtol(rankstr,&end,10);
        if(end==rankstr || *end!=0){ fprintf(stderr,"%s:%d: rank non numerico: '%s'\n",path,lineno,rankstr); exit(1); }
        if(rank!=n){ fprintf(stderr,"%s:%d: rank %ld non contiguo (atteso %d)\n",path,lineno,rank,n); exit(1); }
        unsigned char *bytes; int blen;
        if(!m_b64_decode(b64,&bytes,&blen)){ fprintf(stderr,"%s:%d: base64 non valido: '%s'\n",path,lineno,b64); exit(1); }
        if(n>=cap){ cap = cap? cap*2 : 65536; ents=(MEnt*)realloc(ents,cap*sizeof(MEnt)); }
        ents[n].b=bytes; ents[n].len=blen;
        n++;
    }
    fclose(f);
    T->n_base=n; T->id2b=ents;

    int hcap=1; while(hcap < 2*n) hcap<<=1;
    if(hcap<1) hcap=1;
    hm_init(&T->rank,hcap);
    for(int i=0;i<n;i++) hm_put(&T->rank,(const char*)ents[i].b,ents[i].len,i);

    /* tokenizer_config.json: added_tokens_decoder */
    snprintf(path,sizeof(path),"%s/tokenizer_config.json",model_dir);
    long fn; char *buf=tk_read_file(path,&fn);
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    jval *atd=json_get(root,"added_tokens_decoder");
    if(!atd || atd->t!=J_OBJ){ fprintf(stderr,"%s: manca added_tokens_decoder\n",path); exit(1); }
    T->nsp=atd->len; T->sp=(MSpecial*)calloc(T->nsp?T->nsp:1,sizeof(MSpecial));
    int minsp=INT_MAX;
    for(int i=0;i<atd->len;i++){
        int id=atoi(atd->keys[i]);
        jval *content=json_get(atd->kids[i],"content");
        if(!content || content->t!=J_STR){ fprintf(stderr,"%s: added_tokens_decoder[%s] senza content\n",path,atd->keys[i]); exit(1); }
        T->sp[i].str=content->str; T->sp[i].len=(int)strlen(content->str); T->sp[i].id=id;
        if(id<minsp) minsp=id;
    }
    qsort(T->sp,T->nsp,sizeof(MSpecial),m_cmp_sp_id);
    if(T->nsp>0 && minsp!=T->n_base){
        fprintf(stderr,"%s: id speciale minimo %d != n_base %d\n",path,minsp,T->n_base); exit(1);
    }
    T->n_ids=T->n_base+T->nsp;
}

/* ---------- BPE su un pezzo di byte grezzi (rank-merge tiktoken) ---------- */
static void mbpe_piece(MTok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int n=b-a; if(n<=0) return;
    if(n==1){ int id=hm_get(&T->rank,(const char*)p+a,1); if(id>=0&&*no<max) out[(*no)++]=id; return; }
    /* intero pezzo gia' token: scorciatoia */
    { int id=hm_get(&T->rank,(const char*)p+a,n); if(id>=0){ if(*no<max) out[(*no)++]=id; return; } }
    int *st=(int*)malloc((n+1)*sizeof(int));      /* inizi dei segmenti (relativi ad a), st[k]..st[k+1] */
    for(int i=0;i<=n;i++) st[i]=i; int ns=n;
    for(;;){
        int best=-1, bestrank=INT_MAX;
        for(int i=0;i<ns-1;i++){
            int r=hm_get(&T->rank,(const char*)p+a+st[i],st[i+2]-st[i]);
            if(r>=0 && r<bestrank){ bestrank=r; best=i; }
        }
        if(best<0) break;
        memmove(st+best+1, st+best+2, (ns-best-1)*sizeof(int)); ns--;
    }
    for(int i=0;i<ns;i++){ int id=hm_get(&T->rank,(const char*)p+a+st[i],st[i+1]-st[i]);
        if(id>=0 && *no<max) out[(*no)++]=id; }
    free(st);
}

/* ---------- classi di caratteri del pre-tokenizer moonshot ---------- */
#define M_UPC(c) ((is_LuLt(c)||is_Lm(c)||is_Lo(c)||is_M(c)) && !is_Han(c))
#define M_LOC(c) ((is_Ll(c)||is_Lm(c)||is_Lo(c)||is_M(c)) && !is_Han(c))

static inline int m_isnl(uint32_t c){ return c=='\r'||c=='\n'; }
static inline uint32_t m_low(uint32_t c){ return (c>='A'&&c<='Z') ? c+32 : c; }
/* [^\r\n\p{L}\p{N}]: prefisso opzionale ammesso davanti a U*l+ / U+l* (nota: \p{M} NON e' in \p{L}) */
static inline int m_pre(uint32_t c){ return !is_L(c) && !is_N(c) && !m_isnl(c); }

/* U*l+ con backtracking manuale: U ed l si sovrappongono su Lm/Lo/M.
 * R = massima corsa (UPC|LOC) da i; uend = prefisso massimo di R tutto UPC.
 *   - se uend<|R|: R[uend] e' per costruzione LOC-only -> l-run da uend.
 *   - altrimenti (tutto R e' UPC) se uend>=1 e l'ultimo carattere e' anche
 *     LOC (ambiguo Lm/Lo/M): U* cede l'ultimo carattere a l+ (stesso estremo).
 *   - altrimenti: nessun carattere disponibile per l+, fallisce. */
static int m_match_u_star_l_plus(const uint32_t *cp, int n, int i){
    int rend=i; while(rend<n && (M_UPC(cp[rend])||M_LOC(cp[rend]))) rend++;
    int uend=i; while(uend<rend && M_UPC(cp[uend])) uend++;
    if(uend<rend){ int j=uend; while(j<n && M_LOC(cp[j])) j++; return j; }
    if(uend>i && M_LOC(cp[uend-1])) return uend;
    return i; /* nessun progresso: fallito */
}
/* U+l*: nessuna ambiguita' da risolvere (l* accetta zero occorrenze). */
static int m_match_u_plus_l_star(const uint32_t *cp, int n, int i){
    int j=i; while(j<n && M_UPC(cp[j])) j++;
    if(j==i) return i;
    while(j<n && M_LOC(cp[j])) j++;
    return j;
}
/* (?i:'s|'t|'re|'ve|'m|'ll|'d) come suffisso, dopo un match di U*l+/U+l* */
static int m_match_contraction(const uint32_t *cp, int n, int end){
    if(end<n && cp[end]=='\''){
        if(end+1<n){
            uint32_t d=m_low(cp[end+1]);
            if(end+2<n){
                uint32_t d2=m_low(cp[end+2]);
                if((d=='r'&&d2=='e')||(d=='v'&&d2=='e')||(d=='l'&&d2=='l')) return end+3;
            }
            if(d=='s'||d=='t'||d=='m'||d=='d') return end+2;
        }
    }
    return end;
}
/* alternativa 2/3: prova con prefisso opzionale (greedy, come '?' in regex),
 * poi senza, esattamente come farebbe un motore regex con backtracking. */
static int m_try_alt2(const uint32_t *cp, int n, int i){
    if(i<n && m_pre(cp[i])){
        int m=m_match_u_star_l_plus(cp,n,i+1);
        if(m>i+1) return m_match_contraction(cp,n,m);
    }
    int m=m_match_u_star_l_plus(cp,n,i);
    if(m>i) return m_match_contraction(cp,n,m);
    return -1;
}
static int m_try_alt3(const uint32_t *cp, int n, int i){
    if(i<n && m_pre(cp[i])){
        int m=m_match_u_plus_l_star(cp,n,i+1);
        if(m>i+1) return m_match_contraction(cp,n,m);
    }
    int m=m_match_u_plus_l_star(cp,n,i);
    if(m>i) return m_match_contraction(cp,n,m);
    return -1;
}

/* ---------- pre-tokenizer moonshot su una porzione di testo ---------- */
static void mpretok_chunk(MTok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a; if(nb<=0) return;
    uint32_t *cp=(uint32_t*)malloc((nb+1)*sizeof(uint32_t));
    int *off=(int*)malloc((nb+2)*sizeof(int)); int n=0;
    for(int i=a;i<b;){ uint32_t c; int k=u8_next(p,b,i,&c); off[n]=i; cp[n]=c; n++; i+=k; }
    off[n]=b;
    int i=0;
    while(i<n){
        int start=i;
        /* 1) [\p{Han}]+ */
        if(is_Han(cp[i])){ int j=i; while(j<n && is_Han(cp[j])) j++; i=j; mbpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        /* 2) [pref]? U* l+ (contraz)? */
        { int e=m_try_alt2(cp,n,i); if(e>=0){ i=e; mbpe_piece(T,p,off[start],off[i],out,no,max); continue; } }
        /* 3) [pref]? U+ l* (contraz)? */
        { int e=m_try_alt3(cp,n,i); if(e>=0){ i=e; mbpe_piece(T,p,off[start],off[i],out,no,max); continue; } }
        /* 4) \p{N}{1,3} */
        if(is_N(cp[i])){ int j=i,k=0; while(j<n && is_N(cp[j]) && k<3){ j++; k++; } i=j; mbpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        /* 5)  ?[^\s\p{L}\p{N}]+[\r\n]* */
        {
            int j=i; uint32_t c=cp[i];
            if(c==' ' && j+1<n && !is_S(cp[j+1]) && !is_L(cp[j+1]) && !is_N(cp[j+1])) j++;
            if(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])){
                while(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])) j++;
                while(j<n && m_isnl(cp[j])) j++;
                i=j; mbpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        /* 6) \s*[\r\n]+   7) \s+(?!\S)   8) \s+ */
        {
            int r=i; while(r<n && is_S(cp[r])) r++;
            if(r>i){
                int last=-1; for(int j=i;j<r;j++) if(m_isnl(cp[j])) last=j;
                if(last>=0){ i=last+1; mbpe_piece(T,p,off[start],off[i],out,no,max); continue; }
                int end=(r<n)?r-1:r;
                if(end<=i) end=i+1;
                i=end; mbpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        /* salvagente: non dovrebbe accadere (tutte le classi sono coperte sopra) */
        i++; mbpe_piece(T,p,off[start],off[i],out,no,max);
    }
    free(cp); free(off);
}

/* ---------- encode: testo puro -> id (NESSUNA scansione di special) ---------- */
static int mtok_encode(MTok *T, const char *text, int len, int *out, int max){
    const unsigned char *p=(const unsigned char*)text; int no=0;
    mpretok_chunk(T,p,0,len,out,&no,max);
    return no;
}

/* ---------- id di uno special dato il suo contenuto letterale; -1 se assente ---------- */
static int mtok_special(MTok *T, const char *content){
    for(int i=0;i<T->nsp;i++) if(!strcmp(T->sp[i].str,content)) return T->sp[i].id;
    return -1;
}

/* ---------- decode: id -> byte (id<n_base: byte grezzi; id>=n_base: contenuto letterale dello special) ---------- */
static int mtok_decode(MTok *T, const int *ids, int n, char *out, int max){
    int o=0;
    for(int t=0;t<n;t++){
        int id=ids[t];
        if(id>=0 && id<T->n_base){
            MEnt *e=&T->id2b[id];
            for(int j=0;j<e->len && o<max;j++) out[o++]=(char)e->b[j];
        } else if(id>=T->n_base){
            int lo=0, hi=T->nsp-1, found=-1;
            while(lo<=hi){ int mid=(lo+hi)/2;
                if(T->sp[mid].id==id){ found=mid; break; }
                else if(T->sp[mid].id<id) lo=mid+1; else hi=mid-1; }
            if(found>=0){ const char *s=T->sp[found].str; int l=T->sp[found].len;
                for(int j=0;j<l && o<max;j++) out[o++]=s[j]; }
            /* id sconosciuto: saltato */
        }
        /* id<0: saltato */
    }
    if(o<max) out[o]=0;
    return o;
}

/* ---------- costruttori del chat_template moonshot (Chat-Task 4) ----------
 * Replica fedele di chat_template.jinja per un turno:
 *   <|im_{role}|> + encode(role) + <|im_middle|> + encode(content) + <|im_end|>
 * role_txt/content sono UTF-8 puro (mai special: encode() non li riconosce mai,
 * i soli special immessi sono i marcatori di ruolo, iniettati per id qui).
 * Ritornano il nuovo `no`, o -1 se `max` non basta (il chiamante rifiuta il turno). */
static int mtok_tmpl_msg(MTok *T, const char *role, const char *content, int *out, int no, int max){
    if(no<0) return -1;  /* chiamata incatenata dopo un -1 precedente: propaga il rifiuto, non scrive out[-1] */
    char key[32];
    int kn = snprintf(key,sizeof(key),"<|im_%s|>",role);
    if(kn<0 || kn>=(int)sizeof(key)) return -1;
    int rsp = mtok_special(T,key);
    if(rsp<0) return -1;
    if(no>=max) return -1;
    out[no++]=rsp;

    int rlen=(int)strlen(role);
    if(no+rlen>max) return -1;  /* limite superiore sicuro: mai piu' token di byte grezzi */
    no += mtok_encode(T, role, rlen, out+no, max-no);

    int mid = mtok_special(T,"<|im_middle|>");
    if(mid<0) return -1;
    if(no>=max) return -1;
    out[no++]=mid;

    int clen=(int)strlen(content);
    if(no+clen>max) return -1;
    no += mtok_encode(T, content, clen, out+no, max-no);

    int end = mtok_special(T,"<|im_end|>");
    if(end<0) return -1;
    if(no>=max) return -1;
    out[no++]=end;
    return no;
}
/* prompt di generazione: <|im_assistant|> + encode("assistant") + <|im_middle|> (nessun <|im_end|>:
 * il modello continua da qui). */
static int mtok_tmpl_genprompt(MTok *T, int *out, int no, int max){
    if(no<0) return -1;  /* stessa propagazione difensiva di mtok_tmpl_msg */
    int asp = mtok_special(T,"<|im_assistant|>");
    if(asp<0) return -1;
    if(no>=max) return -1;
    out[no++]=asp;

    static const char role[]="assistant";
    int rlen=(int)(sizeof(role)-1);
    if(no+rlen>max) return -1;
    no += mtok_encode(T, role, rlen, out+no, max-no);

    int mid = mtok_special(T,"<|im_middle|>");
    if(mid<0) return -1;
    if(no>=max) return -1;
    out[no++]=mid;
    return no;
}

#undef M_UPC
#undef M_LOC

#endif
