/*
 * kernels.c — TurboQuantCPU: SIMD-accelerated hot paths
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC \
 *       -o _tqcpu_kernels.so kernels.c -lm
 *
 * SIMD hierarchy (runtime CPUID dispatch):
 *   x86: AVX-512F+VNNI → AVX2+FMA → SSE4.1 → scalar
 *   ARM: NEON (always on armv8) → scalar
 *
 * Hot paths:
 *   tqcpu_fwht_f32            - normalised FWHT, single vector
 *   tqcpu_rfwht_f32           - randomised FWHT (signs * x, then FWHT)
 *   tqcpu_fwht_batch_f32      - batch FWHT, B×d row-major (OpenMP)
 *   tqcpu_rfwht_batch_f32     - batch randomised FWHT (OpenMP)
 *   tqcpu_sign_i8             - sign-quantize f32 → i8 {-1,+1}
 *   tqcpu_pack_signs          - i8 {-1,+1} → 1-bit/elem packed uint8
 *   tqcpu_unpack_signs        - packed uint8 → i8 {-1,+1}
 *   tqcpu_dot_f32_i8          - f32 · i8 → f32
 *   tqcpu_matmul_f32_i8       - (B×d f32) @ (M×d i8)^T → (B×M f32)
 *   tqcpu_quantize_lloyd      - scalar Lloyd-Max quantize f32 → uint8
 *   tqcpu_dequantize_lloyd    - uint8 → f32 via centroid table
 *   tqcpu_polar_encode        - polar encode 2-D pairs → (r, θ) uint8
 *   tqcpu_polar_dot           - table-lookup dot for PolarQuant
 *   tqcpu_binary_dot          - popcount XOR dot (packed bits)
 *   tqcpu_softmax_f32         - numerically-stable softmax in-place
 *   tqcpu_weighted_sum_f32    - weighted sum of rows (softmax attention output)
 *   tqcpu_version             - version string
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* ─── CPUID detection ─────────────────────────────────────────────── */

static int g_simd_level = -1;
/* 0=scalar 1=sse41 2=avx2 3=avx512f 4=avx512vnni */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <cpuid.h>
static void _detect_simd(void) {
    if (g_simd_level >= 0) return;
    g_simd_level = 0;
    unsigned eax=0, ebx=0, ecx=0, edx=0;
    __cpuid(1, eax, ebx, ecx, edx);
    if (ecx & (1u<<19)) g_simd_level = 1;  /* SSE4.1 */
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (ebx & (1u<<5))  g_simd_level = 2;  /* AVX2   */
    if (ebx & (1u<<16)) g_simd_level = 3;  /* AVX512F */
    if (ecx & (1u<<11)) g_simd_level = 4;  /* AVX512-VNNI */
}
#else
static void _detect_simd(void) { g_simd_level = 0; }
#endif

/* ─── Scalar FWHT ─────────────────────────────────────────────────── */

static void _fwht_scalar(float *x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2*h) {
            for (int j = i; j < i+h; j++) {
                float lo = x[j], hi = x[j+h];
                x[j]   = lo + hi;
                x[j+h] = lo - hi;
            }
        }
    }
    float s = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= s;
}

/* ─── AVX2 FWHT ───────────────────────────────────────────────────── */

#if defined(__AVX2__)
#include <immintrin.h>

static void _fwht_avx2(float *x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        if (h >= 8) {
            for (int i = 0; i < n; i += 2*h) {
                float *lo = x+i, *hi = x+i+h;
                int j = 0;
                for (; j+8 <= h; j += 8) {
                    __m256 va = _mm256_loadu_ps(lo+j);
                    __m256 vb = _mm256_loadu_ps(hi+j);
                    _mm256_storeu_ps(lo+j, _mm256_add_ps(va,vb));
                    _mm256_storeu_ps(hi+j, _mm256_sub_ps(va,vb));
                }
                for (; j < h; j++) {
                    float a=lo[j], b=hi[j];
                    lo[j]=a+b; hi[j]=a-b;
                }
            }
        } else {
            for (int i = 0; i < n; i += 2*h)
                for (int j = i; j < i+h; j++) {
                    float a=x[j], b=x[j+h];
                    x[j]=a+b; x[j+h]=a-b;
                }
        }
    }
    float s = 1.0f / sqrtf((float)n);
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i+8 <= n; i+=8)
        _mm256_storeu_ps(x+i, _mm256_mul_ps(_mm256_loadu_ps(x+i), vs));
    for (; i < n; i++) x[i] *= s;
}

/* AVX2 FMA dot: f32 · i8 → f32 */
static float _dot_f32_i8_avx2(const float *a, const int8_t *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i+8 <= n; i+=8) {
        __m256 va  = _mm256_loadu_ps(a+i);
        __m128i vb8 = _mm_loadl_epi64((const __m128i*)(b+i));
        __m256i vb32 = _mm256_cvtepi8_epi32(vb8);
        __m256 vbf  = _mm256_cvtepi32_ps(vb32);
        acc = _mm256_fmadd_ps(va, vbf, acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc,1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 s  = _mm_add_ps(hi,lo);
    s = _mm_hadd_ps(s,s); s = _mm_hadd_ps(s,s);
    float r = _mm_cvtss_f32(s);
    for (; i < n; i++) r += a[i]*(float)b[i];
    return r;
}

/* AVX2 sign-quantize: f32 → i8 {-1,+1} */
static void _sign_i8_avx2(const float *x, int8_t *y, int n) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i+8 <= n; i+=8) {
        __m256 v   = _mm256_loadu_ps(x+i);
        /* cmp_ge: 0xFFFFFFFF if x>=0, else 0 */
        __m256 pos = _mm256_cmp_ps(v, zero, _CMP_GE_OQ);
        /* pos ? 1.0 : -1.0 */
        __m256 res = _mm256_blendv_ps(_mm256_set1_ps(-1.0f),
                                       _mm256_set1_ps( 1.0f), pos);
        /* convert to int8 via int32 */
        __m256i ri = _mm256_cvtps_epi32(res);
        __m128i lo128 = _mm256_castsi256_si128(ri);
        __m128i hi128 = _mm256_extracti128_si256(ri,1);
        __m128i p16 = _mm_packs_epi32(lo128, hi128);
        __m128i p8  = _mm_packs_epi16(p16, _mm_setzero_si128());
        _mm_storel_epi64((__m128i*)(y+i), p8);
    }
    for (; i < n; i++) y[i] = (x[i] >= 0.0f) ? 1 : -1;
}

#endif /* __AVX2__ */

/* ─── AVX-512F FWHT ───────────────────────────────────────────────── */

#if defined(__AVX512F__)
#include <immintrin.h>

static void _fwht_avx512(float *x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        if (h >= 16) {
            for (int i = 0; i < n; i += 2*h) {
                float *lo=x+i, *hi=x+i+h;
                int j=0;
                for (; j+16 <= h; j+=16) {
                    __m512 va=_mm512_loadu_ps(lo+j);
                    __m512 vb=_mm512_loadu_ps(hi+j);
                    _mm512_storeu_ps(lo+j, _mm512_add_ps(va,vb));
                    _mm512_storeu_ps(hi+j, _mm512_sub_ps(va,vb));
                }
                /* 8-wide tail */
#if defined(__AVX2__)
                for (; j+8 <= h; j+=8) {
                    __m256 va=_mm256_loadu_ps(lo+j);
                    __m256 vb=_mm256_loadu_ps(hi+j);
                    _mm256_storeu_ps(lo+j, _mm256_add_ps(va,vb));
                    _mm256_storeu_ps(hi+j, _mm256_sub_ps(va,vb));
                }
#endif
                for (; j < h; j++) {
                    float a=lo[j],b=hi[j];
                    lo[j]=a+b; hi[j]=a-b;
                }
            }
        } else {
            for (int i=0; i<n; i+=2*h)
                for (int j=i; j<i+h; j++) {
                    float a=x[j],b=x[j+h];
                    x[j]=a+b; x[j+h]=a-b;
                }
        }
    }
    float s = 1.0f/sqrtf((float)n);
    __m512 vs = _mm512_set1_ps(s);
    int i=0;
    for (; i+16<=n; i+=16)
        _mm512_storeu_ps(x+i, _mm512_mul_ps(_mm512_loadu_ps(x+i),vs));
    for (; i<n; i++) x[i]*=s;
}

#endif /* __AVX512F__ */

/* ─── ARM NEON FWHT ───────────────────────────────────────────────── */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

static void _fwht_neon(float *x, int n) {
    for (int h=1; h<n; h<<=1) {
        if (h >= 4) {
            for (int i=0; i<n; i+=2*h) {
                float *lo=x+i, *hi=x+i+h;
                int j=0;
                for (; j+4<=h; j+=4) {
                    float32x4_t va=vld1q_f32(lo+j);
                    float32x4_t vb=vld1q_f32(hi+j);
                    vst1q_f32(lo+j, vaddq_f32(va,vb));
                    vst1q_f32(hi+j, vsubq_f32(va,vb));
                }
                for (; j<h; j++) {
                    float a=lo[j],b=hi[j];
                    lo[j]=a+b; hi[j]=a-b;
                }
            }
        } else {
            for (int i=0; i<n; i+=2*h)
                for (int j=i; j<i+h; j++) {
                    float a=x[j],b=x[j+h];
                    x[j]=a+b; x[j+h]=a-b;
                }
        }
    }
    float s = 1.0f/sqrtf((float)n);
    float32x4_t vs = vdupq_n_f32(s);
    int i=0;
    for (; i+4<=n; i+=4)
        vst1q_f32(x+i, vmulq_f32(vld1q_f32(x+i),vs));
    for (; i<n; i++) x[i]*=s;
}

static float _dot_f32_i8_neon(const float *a, const int8_t *b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i=0;
    for (; i+4<=n; i+=4) {
        float32x4_t va = vld1q_f32(a+i);
        int8x8_t    vb8 = vld1_s8(b+i);
        int16x8_t   vb16 = vmovl_s8(vb8);
        int32x4_t   vb32 = vmovl_s16(vget_low_s16(vb16));
        float32x4_t vbf  = vcvtq_f32_s32(vb32);
        acc = vmlaq_f32(acc, va, vbf);
    }
    float32x2_t s2 = vadd_f32(vget_high_f32(acc), vget_low_f32(acc));
    float r = vget_lane_f32(vpadd_f32(s2,s2), 0);
    for (; i<n; i++) r += a[i]*(float)b[i];
    return r;
}

#endif /* __ARM_NEON */

/* ─── Dispatch wrappers (exported) ────────────────────────────────── */

void tqcpu_fwht_f32(float *x, int n) {
    _detect_simd();
#if defined(__AVX512F__)
    if (g_simd_level >= 3) { _fwht_avx512(x,n); return; }
#endif
#if defined(__AVX2__)
    if (g_simd_level >= 2) { _fwht_avx2(x,n); return; }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    _fwht_neon(x,n); return;
#endif
    _fwht_scalar(x,n);
}

void tqcpu_rfwht_f32(float *x, const float *signs, int n) {
    /* x[i] *= signs[i], then FWHT */
#if defined(__AVX2__)
    _detect_simd();
    if (g_simd_level >= 2) {
        int i=0;
        for (; i+8<=n; i+=8) {
            __m256 vx = _mm256_loadu_ps(x+i);
            __m256 vs = _mm256_loadu_ps(signs+i);
            _mm256_storeu_ps(x+i, _mm256_mul_ps(vx,vs));
        }
        for (; i<n; i++) x[i]*=signs[i];
    } else
#endif
    { for (int i=0; i<n; i++) x[i]*=signs[i]; }
    tqcpu_fwht_f32(x,n);
}

void tqcpu_fwht_batch_f32(float *X, int B, int n) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (int b=0; b<B; b++)
        tqcpu_fwht_f32(X + (long long)b*n, n);
#else
    for (int b=0; b<B; b++)
        tqcpu_fwht_f32(X + (long long)b*n, n);
#endif
}

void tqcpu_rfwht_batch_f32(float *X, const float *signs, int B, int n) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (int b=0; b<B; b++) {
        float *xb = X + (long long)b*n;
        for (int i=0; i<n; i++) xb[i]*=signs[i];
        tqcpu_fwht_f32(xb, n);
    }
#else
    for (int b=0; b<B; b++) {
        float *xb = X + (long long)b*n;
        for (int i=0; i<n; i++) xb[i]*=signs[i];
        tqcpu_fwht_f32(xb, n);
    }
#endif
}

void tqcpu_sign_i8(const float *x, int8_t *y, int n) {
#if defined(__AVX2__)
    _detect_simd();
    if (g_simd_level >= 2) { _sign_i8_avx2(x,y,n); return; }
#endif
    for (int i=0; i<n; i++) y[i] = (x[i]>=0.0f) ? 1 : -1;
}

void tqcpu_pack_signs(const int8_t *signs, uint8_t *out, int n) {
    int nb = (n+7)/8;
    memset(out, 0, nb);
    for (int i=0; i<n; i++)
        if (signs[i]>0) out[i>>3] |= (uint8_t)(1u<<(i&7));
}

void tqcpu_unpack_signs(const uint8_t *packed, int8_t *out, int n) {
    for (int i=0; i<n; i++)
        out[i] = ((packed[i>>3]>>(i&7))&1) ? 1 : -1;
}

float tqcpu_dot_f32_i8(const float *a, const int8_t *b, int n) {
    _detect_simd();
#if defined(__AVX2__)
    if (g_simd_level >= 2) return _dot_f32_i8_avx2(a,b,n);
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return _dot_f32_i8_neon(a,b,n);
#endif
    float s=0; for (int i=0; i<n; i++) s+=a[i]*(float)b[i]; return s;
}

/*
 * (B×n f32) @ (M×n i8)^T → (B×M f32)
 * OpenMP parallel over M (outer loop), SIMD inner dot.
 */
void tqcpu_matmul_f32_i8(
    const float  *A,   /* B×n */
    const int8_t *B,   /* M×n */
    float        *C,   /* B×M */
    int batch, int M, int n
) {
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b=0; b<batch; b++)
        for (int m=0; m<M; m++)
            C[b*M+m] = tqcpu_dot_f32_i8(A+b*n, B+(long long)m*n, n);
#else
    for (int b=0; b<batch; b++)
        for (int m=0; m<M; m++)
            C[b*M+m] = tqcpu_dot_f32_i8(A+b*n, B+(long long)m*n, n);
#endif
}

/*
 * Lloyd-Max scalar quantize: for each element x[i], find index j such that
 * boundaries[j-1] < x[i] <= boundaries[j]  (boundaries has n_levels-1 entries)
 * Vectorised binary search via linear scan (n_levels ≤ 256, cache-friendly).
 */
void tqcpu_quantize_lloyd(
    const float   *x,
    uint8_t       *out,
    int            n,
    const float   *boundaries,  /* n_levels-1 values, ascending */
    int            n_levels
) {
    int nb = n_levels-1;
#if defined(__AVX2__)
    _detect_simd();
    if (g_simd_level >= 2 && nb <= 255) {
        int i=0;
        for (; i+8<=n; i+=8) {
            __m256 vx = _mm256_loadu_ps(x+i);
            __m256i idx = _mm256_setzero_si256();
            /* count how many boundaries x exceeds */
            for (int k=0; k<nb; k++) {
                __m256 vb = _mm256_set1_ps(boundaries[k]);
                /* mask: 1 where x > b */
                __m256 mask = _mm256_cmp_ps(vx, vb, _CMP_GT_OQ);
                __m256i one = _mm256_set1_epi32(1);
                idx = _mm256_add_epi32(idx,
                      _mm256_and_si256(_mm256_castps_si256(mask), one));
            }
            /* extract and store 8 bytes */
            int32_t tmp[8];
            _mm256_storeu_si256((__m256i*)tmp, idx);
            for (int k=0; k<8; k++) out[i+k] = (uint8_t)tmp[k];
        }
        for (; i<n; i++) {
            int j=0;
            while (j<nb && x[i]>boundaries[j]) j++;
            out[i] = (uint8_t)j;
        }
        return;
    }
#endif
    for (int i=0; i<n; i++) {
        int j=0;
        while (j<nb && x[i]>boundaries[j]) j++;
        out[i] = (uint8_t)j;
    }
}

void tqcpu_dequantize_lloyd(
    const uint8_t *idx,
    float         *out,
    int            n,
    const float   *levels   /* n_levels centroids */
) {
    for (int i=0; i<n; i++) out[i] = levels[idx[i]];
}

/*
 * Polar encode: for each pair (x[2i], x[2i+1]) compute
 *   r = sqrt(x^2 + y^2)
 *   theta = atan2(y, x)  mapped to [0, 2pi)
 * then quantize r and theta using the provided codebooks.
 * Outputs:
 *   r_idx[i]     : quantized radius index (uint8)
 *   theta_idx[i] : quantized angle index  (uint8)
 */
void tqcpu_polar_encode(
    const float *x,           /* n elements: n/2 pairs */
    uint8_t     *r_idx,
    uint8_t     *theta_idx,
    int          n_pairs,
    const float *r_bounds,   int n_r_levels,
    const float *theta_bounds, int n_theta_levels
) {
    int nr = n_r_levels-1;
    int nt = n_theta_levels-1;
    for (int i=0; i<n_pairs; i++) {
        float xi = x[2*i], yi = x[2*i+1];
        float r  = sqrtf(xi*xi + yi*yi);
        float th = atan2f(yi, xi);
        if (th < 0) th += 6.28318530718f;  /* map [-pi,pi] → [0,2pi) */

        /* quantize r */
        int jr=0; while (jr<nr && r>r_bounds[jr]) jr++;
        r_idx[i] = (uint8_t)jr;

        /* quantize theta */
        int jt=0; while (jt<nt && th>theta_bounds[jt]) jt++;
        theta_idx[i] = (uint8_t)jt;
    }
}

/*
 * Polar inner-product via table lookup.
 * For each cached key (r_i, theta_i) and query pair (qx_i, qy_i):
 *   <q_pair, k_pair> ≈ r_i * (qx_i*cos(theta_i) + qy_i*sin(theta_i))
 * Precomputed cos/sin table provided (n_theta_levels entries).
 */
float tqcpu_polar_dot_seq(
    const float   *q,            /* 2*n_pairs query floats */
    const uint8_t *r_idx,        /* n_pairs radius indices */
    const uint8_t *theta_idx,    /* n_pairs theta indices  */
    const float   *r_levels,
    const float   *cos_table,
    const float   *sin_table,
    int            n_pairs
) {
    float acc = 0.0f;
    for (int i=0; i<n_pairs; i++) {
        float ri  = r_levels[r_idx[i]];
        float ci  = cos_table[theta_idx[i]];
        float si  = sin_table[theta_idx[i]];
        acc += ri * (q[2*i]*ci + q[2*i+1]*si);
    }
    return acc;
}

/*
 * Batch polar dot: (n_seq polar keys) with (1 query) → n_seq scores
 * r_idx, theta_idx: (n_seq × n_pairs) row-major
 * q:                (2*n_pairs) query vector
 */
void tqcpu_polar_dot_batch(
    const float   *q,            /* 2*n_pairs */
    const uint8_t *r_idx,        /* n_seq × n_pairs */
    const uint8_t *theta_idx,    /* n_seq × n_pairs */
    float         *out,          /* n_seq scores */
    const float   *r_levels,
    const float   *cos_table,
    const float   *sin_table,
    int            n_seq,
    int            n_pairs
) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (int s=0; s<n_seq; s++) {
        out[s] = tqcpu_polar_dot_seq(
            q,
            r_idx   + (long long)s*n_pairs,
            theta_idx + (long long)s*n_pairs,
            r_levels, cos_table, sin_table,
            n_pairs
        );
    }
#else
    for (int s=0; s<n_seq; s++) {
        out[s] = tqcpu_polar_dot_seq(
            q,
            r_idx   + (long long)s*n_pairs,
            theta_idx + (long long)s*n_pairs,
            r_levels, cos_table, sin_table,
            n_pairs
        );
    }
#endif
}

/*
 * Binary (popcount) dot: <x,y> in {-1,+1}^n via packed uint8.
 * = n - 2 * popcount(x XOR y)
 */
int tqcpu_binary_dot(const uint8_t *x, const uint8_t *y, int nbytes) {
    int xored=0, i=0;
#if defined(__POPCNT__)
    for (; i+8<=nbytes; i+=8) {
        uint64_t xv, yv;
        memcpy(&xv, x+i, 8); memcpy(&yv, y+i, 8);
        xored += __builtin_popcountll(xv^yv);
    }
#endif
    for (; i<nbytes; i++) xored += __builtin_popcount((unsigned)(x[i]^y[i]));
    return nbytes*8 - 2*xored;
}

/*
 * Numerically-stable softmax, in-place.
 * x[0..n-1] → softmax(x)[0..n-1]
 */
void tqcpu_softmax_f32(float *x, int n) {
    float mx = x[0];
    for (int i=1; i<n; i++) if (x[i]>mx) mx=x[i];
    float sum=0;
    for (int i=0; i<n; i++) { x[i]=expf(x[i]-mx); sum+=x[i]; }
    float inv = 1.0f/(sum+1e-12f);
    for (int i=0; i<n; i++) x[i]*=inv;
}

/*
 * Weighted sum of rows: out[j] = sum_s w[s] * V[s, j]
 * V: (n_seq × d) row-major float32
 * w: (n_seq,) weights (after softmax)
 * out: (d,) result
 */
void tqcpu_weighted_sum_f32(
    const float *w,    /* n_seq */
    const float *V,    /* n_seq × d */
    float       *out,  /* d */
    int          n_seq,
    int          d
) {
    memset(out, 0, d*sizeof(float));
#if defined(__AVX2__)
    _detect_simd();
    if (g_simd_level >= 2) {
        for (int s=0; s<n_seq; s++) {
            __m256 vw = _mm256_set1_ps(w[s]);
            const float *vs = V + (long long)s*d;
            int j=0;
            for (; j+8<=d; j+=8) {
                __m256 vo = _mm256_loadu_ps(out+j);
                __m256 vv = _mm256_loadu_ps(vs+j);
                _mm256_storeu_ps(out+j, _mm256_fmadd_ps(vw,vv,vo));
            }
            for (; j<d; j++) out[j]+=w[s]*vs[j];
        }
        return;
    }
#endif
    for (int s=0; s<n_seq; s++) {
        const float *vs = V + (long long)s*d;
        for (int j=0; j<d; j++) out[j]+=w[s]*vs[j];
    }
}

/* ─── Version ─────────────────────────────────────────────────────── */

const char *tqcpu_version(void) {
    _detect_simd();
    static char buf[128];
    const char *lvl="scalar";
    if (g_simd_level>=4) lvl="avx512-vnni";
    else if (g_simd_level>=3) lvl="avx512f";
    else if (g_simd_level>=2) lvl="avx2+fma";
    else if (g_simd_level>=1) lvl="sse4.1";
#if defined(__ARM_NEON)
    lvl="neon";
#endif
#ifdef _OPENMP
    snprintf(buf, sizeof(buf),
        "tqcpu_kernels v0.0.10 [%s] OpenMP=%d", lvl, omp_get_max_threads());
#else
    snprintf(buf, sizeof(buf), "tqcpu_kernels v0.0.10 [%s] no-OpenMP", lvl);
#endif
    return buf;
}