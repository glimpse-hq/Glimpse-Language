// Kernels for Ternary Bonsai 2 (Qwen3.5 hybrid, 2B and 4B). Text activations are fp32, like PrismML's MLX runtime.
// Packed linear: y = Wq @ fwht512(signs * x), Wq = scale * (code - 1), codes 2-bit in u32 (lane j at bits 2j).
// Sizes come from the host (config.json). Fixed by the architecture: Hadamard block 512, GDN head dim 128,
// attention head dim 256, conv kernel 4, 4 q heads per kv head.
#include <metal_stdlib>
using namespace metal;

constant float INV_SQRT512 = 0.044194173824159216f;
constant float EPS = 1e-6f;
// Rotation kernels work on chunks of 4 Hadamard blocks; grid.y = chunk index.
constant uint CHUNK = 2048;

// Block reduction helper: sum over the threadgroup. red must hold 32 floats.
inline float tg_sum(float v, threadgroup float* red, uint sid, uint lane, uint nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? red[lane] : 0.0f;
    return simd_sum(r);
}

inline float tg_max(float v, threadgroup float* red, uint sid, uint lane, uint nsg) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? red[lane] : -INFINITY;
    return simd_max(r);
}

// In-place unnormalized Walsh-Hadamard transform over width/512 blocks of 512 in threadgroup memory.
inline void fwht_tg(threadgroup float* x, uint width, uint tid, uint nt) {
    uint pairs = width / 2;
    for (uint h = 1; h < 512; h <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = tid; p < pairs; p += nt) {
            uint blk = p >> 8, q = p & 255;
            uint i = (blk << 9) + (q / h) * 2 * h + (q % h);
            float a = x[i], b = x[i + h];
            x[i] = a + b;
            x[i + h] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Fused LoRA A: rotation kernels get extra threadgroups past the rotation chunks (grid.y >= nch), each serving
// nsg / 4 rows of u[t, row] = A[row] . x for the unrotated input x, 4 simdgroups per row, 128 values per
// simdgroup step (one GDN head). R = 0: no extra threadgroups. lred: 32 floats of threadgroup memory.
#define LORA_ROW(nch, expr)                                                                     \
    if (tg.y >= nch) {                                                                          \
        uint per = nt / 128, rl = sid / 4, pi = sid % 4, row = (tg.y - nch) * per + rl;         \
        float acc = 0;                                                                          \
        if (row < R) {                                                                          \
            device const half* a = A + row * K;                                                 \
            for (uint k = (pi * 32 + lane) * 4; k < K; k += 512)                                \
                acc += dot(float4(*(device const half4*)(a + k)), (expr));                      \
        }                                                                                       \
        acc = simd_sum(acc);                                                                    \
        if (lane == 0) lred[sid] = acc;                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
        if (row < R && pi == 0 && lane == 0)                                                    \
            U[t * R + row] = lred[sid] + lred[sid + 1] + lred[sid + 2] + lred[sid + 3];         \
        return;                                                                                 \
    }

inline float4 f4(device const float* p) { return *(device const float4*)p; }
inline float4 silu4(float4 a) { return a / (1.0f + exp(-a)); }

// h[t] -> xn = rmsnorm(h) * w (optional), xr = fwht(signs * xn) / sqrt(512). Grid (T, ceil(D / CHUNK) + LoRA),
// 512 threads. Every chunk computes the full-row norm in the same order, so results do not depend on the chunking.
kernel void norm_rotate(device const float* h [[buffer(0)]], device const float* w [[buffer(1)]],
                        device const float* signs [[buffer(2)]], device float* xn [[buffer(3)]],
                        device float* xr [[buffer(4)]], constant uint& D [[buffer(5)]],
                        constant uint& write_xn [[buffer(6)]], device const half* A [[buffer(7)]],
                        device float* U [[buffer(8)]], constant uint& R [[buffer(9)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint2 ntp [[threads_per_threadgroup]],
                        uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[CHUNK];
    threadgroup float red[32];
    threadgroup float lred[32];
    uint K = D, nt = ntp.x, t = tg.x, nch = (D + CHUNK - 1) / CHUNK;
    device const float* x = h + t * D;
    float ss = 0;
    for (uint i = tid; i < D; i += nt) { float v = x[i]; ss += v * v; }
    float r = rsqrt(tg_sum(ss, red, sid, lane, nt / 32) / float(D) + EPS);
    LORA_ROW(nch, f4(x + k) * r * f4(w + k))
    uint c0 = tg.y * CHUNK, n = min(CHUNK, D - c0);
    for (uint i = tid; i < n; i += nt) {
        float v = x[c0 + i] * r * w[c0 + i];
        if (write_xn) xn[t * D + c0 + i] = v;
        buf[i] = v * signs[c0 + i];
    }
    fwht_tg(buf, n, tid, nt);
    for (uint i = tid; i < n; i += nt) xr[t * D + c0 + i] = buf[i] * INV_SQRT512;
}

// xr = fwht(signs * silu(g) * u) (MLP down input).
kernel void swiglu_rotate(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                          device const float* signs [[buffer(2)]], device float* xr [[buffer(3)]],
                          constant uint& D [[buffer(4)]], device const half* A [[buffer(5)]],
                          device float* U [[buffer(6)]], constant uint& R [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint2 ntp [[threads_per_threadgroup]], uint sid [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[CHUNK];
    threadgroup float lred[32];
    uint K = D, nt = ntp.x, t = tg.x, nch = (D + CHUNK - 1) / CHUNK;
    LORA_ROW(nch, silu4(f4(g + t * D + k)) * f4(u + t * D + k))
    uint c0 = tg.y * CHUNK, n = min(CHUNK, D - c0);
    for (uint i = tid; i < n; i += nt) {
        uint e = t * D + c0 + i;
        float a = g[e];
        buf[i] = (a / (1.0f + exp(-a))) * u[e] * signs[c0 + i];
    }
    fwht_tg(buf, n, tid, nt);
    for (uint i = tid; i < n; i += nt) xr[t * D + c0 + i] = buf[i] * INV_SQRT512;
}

inline float4 sigmoid4(float4 a) { return 1.0f / (1.0f + exp(-a)); }

// Attention output gate then rotate: x[e] = attn[e] * sigmoid(gate), gate from q_proj output (per head 256 q,
// 256 gate). W = heads * 256.
kernel void gate_rotate(device const float* attn [[buffer(0)]], device const float* qp [[buffer(1)]],
                        device const float* signs [[buffer(2)]], device float* xr [[buffer(3)]],
                        constant uint& W [[buffer(4)]], device const half* A [[buffer(5)]],
                        device float* U [[buffer(6)]], constant uint& R [[buffer(7)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint2 ntp [[threads_per_threadgroup]], uint sid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[CHUNK];
    threadgroup float lred[32];
    uint K = W, nt = ntp.x, t = tg.x, nch = (W + CHUNK - 1) / CHUNK;
    LORA_ROW(nch, f4(attn + t * W + k) * sigmoid4(f4(qp + t * 2 * W + (k >> 8) * 512 + 256 + (k & 255))))
    uint c0 = tg.y * CHUNK, n = min(CHUNK, W - c0);
    for (uint i = tid; i < n; i += nt) {
        uint e = c0 + i, hd = e >> 8, d = e & 255;
        float gt = qp[t * 2 * W + hd * 512 + 256 + d];
        buf[i] = attn[t * W + e] * (1.0f / (1.0f + exp(-gt))) * signs[e];
    }
    fwht_tg(buf, n, tid, nt);
    for (uint i = tid; i < n; i += nt) xr[t * W + c0 + i] = buf[i] * INV_SQRT512;
}

// GDN head output for the 4 values at k (one simdgroup covers a 128-wide head): rmsnorm(y) * w * silu(z).
inline float4 gated4(device const float* y, device const float* z, device const float* w, uint k) {
    float4 v = f4(y + k);
    float r = rsqrt(simd_sum(dot(v, v)) / 128.0f + EPS);
    return silu4(f4(z + k)) * (v * r * f4(w + (k & 127)));
}

// GDN output: per head (128) rmsnorm(y) * w * silu(z), then rotate. W = v heads * 128 (multiple of CHUNK).
// 512 threads x 4 values per chunk, simdgroup = head.
kernel void gated_norm_rotate(device const float* y [[buffer(0)]], device const float* z [[buffer(1)]],
                              device const float* w [[buffer(2)]], device const float* signs [[buffer(3)]],
                              device float* xr [[buffer(4)]], constant uint& W [[buffer(5)]],
                              device const half* A [[buffer(6)]], device float* U [[buffer(7)]],
                              constant uint& R [[buffer(8)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float buf[CHUNK];
    threadgroup float lred[32];
    uint K = W, nt = 512, t = tg.x, nch = W / CHUNK;
    LORA_ROW(nch, gated4(y + t * W, z + t * W, w, k))
    uint c0 = tg.y * CHUNK;
    uint e0 = tid * 4;
    float4 v = *(device const float4*)(y + t * W + c0 + e0);
    float r = rsqrt(simd_sum(dot(v, v)) / 128.0f + EPS);
    float4 zz = *(device const float4*)(z + t * W + c0 + e0);
    for (uint j = 0; j < 4; j++) {
        uint e = e0 + j;
        float x = v[j] * r * w[e & 127];
        float a = zz[j];
        buf[e] = (a / (1.0f + exp(-a))) * x * signs[c0 + e];
    }
    fwht_tg(buf, CHUNK, tid, 512);
    for (uint j = 0; j < 4; j++) xr[t * W + c0 + e0 + j] = buf[e0 + j] * INV_SQRT512;
}

// Embedding: row = (code - 1) * scale (fp16), inverse transform: fwht then * signs, rounded to fp16.
// Grid (T, ceil(D / CHUNK)).
kernel void embed(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                  device const float* signs [[buffer(2)]], device const int* ids [[buffer(3)]],
                  device float* out [[buffer(4)]], constant uint& D [[buffer(5)]],
                  uint2 tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]], uint2 ntp [[threads_per_threadgroup]]) {
    threadgroup float buf[CHUNK];
    uint t = tg.x, c0 = tg.y * CHUNK, n = min(CHUNK, D - c0), nt = ntp.x;
    uint row = uint(ids[t]);
    uint kw = D / 16, kg = D / 128;
    for (uint i = tid; i < n; i += nt) {
        uint k = c0 + i;
        uint wd = W[row * kw + (k >> 4)];
        float c = float((wd >> (2 * (k & 15))) & 3u) - 1.0f;
        buf[i] = float(half(c * float(S[row * kg + (k >> 7)])));
    }
    fwht_tg(buf, n, tid, nt);
    for (uint i = tid; i < n; i += nt) out[t * D + c0 + i] = float(half(buf[i] * INV_SQRT512 * signs[c0 + i]));
}

// Shared matvec body: R rows from row0, 16 codes per lane per step (fp16-trick decode, x pre-scaled per slot).
// LoRA epilogue when lr > 0: y[row] += Bm[row, 0..lr) . u (Bm fp16 [rows, lr], alpha / r folded in).
template <int R>
inline void qmv_rows(device const uint* W, device const half* S, device const float* x, device float* y,
                     uint K, uint row0, uint add, uint lane, device const half* Bm, device const float* u, uint lr) {
    uint kw = K / 16, kg = K / 128;
    float acc[R];
    for (int r = 0; r < R; r++) acc[r] = 0;
    for (uint k0 = lane * 16; k0 < K; k0 += 512) {
        float4 a0 = *(device const float4*)(x + k0), a1 = *(device const float4*)(x + k0 + 4);
        float4 b0 = *(device const float4*)(x + k0 + 8), b1 = *(device const float4*)(x + k0 + 12);
        float lo[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        float hi[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        float2 xv[8];
        float xs = 0;
        for (uint j = 0; j < 8; j++) {
            float sc = j < 5 ? exp2(-2.0f * float(j)) : exp2(-2.0f * float(j - 5));
            xs += lo[j] + hi[j];
            xv[j] = float2(lo[j], hi[j]) * sc;
        }
        for (int r = 0; r < R; r++) {
            uint wd = W[(row0 + r) * kw + k0 / 16];
            uint lo2 = wd, hi2 = wd >> 10;
            float d = 0;
            for (uint j = 0; j < 8; j++) {
                uint m = j < 5 ? (lo2 & (0x00030003u << (2 * j))) : (hi2 & (0x00030003u << (2 * (j - 5))));
                float2 f = float2(as_type<half2>(m | 0x64006400u) - half2(1024.0h));
                d += xv[j].x * f.x + xv[j].y * f.y;
            }
            acc[r] += float(S[(row0 + r) * kg + k0 / 128]) * (d - xs);
        }
    }
    for (uint j = lane; j < lr; j += 32) {
        float uj = u[j];
        for (int r = 0; r < R; r++) acc[r] += uj * float(Bm[(row0 + r) * lr + j]);
    }
    for (int r = 0; r < R; r++) {
        float v = simd_sum(acc[r]);
        if (lane == 0) y[row0 + r] = add ? y[row0 + r] + v : v;
    }
}

// Decode matvec (one token), 2 simdgroups x 4 rows per threadgroup. Optional LoRA epilogue (B, u, lr).
kernel void qmv4(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                 device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                 constant uint& K [[buffer(4)]], constant uint& O [[buffer(5)]], constant uint& add [[buffer(6)]],
                 device const half* B [[buffer(7)]], device const float* u [[buffer(8)]], constant uint& lr [[buffer(9)]],
                 uint tg [[threadgroup_position_in_grid]], uint sid [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    qmv_rows<4>(W, S, x, y, K, (tg * 2 + sid) * 4, add, lane, B, u, lr);
}

// T-row matvec for verify (T <= NT): qmv4's loop with each row's codes decoded once and applied to every token.
// 2 simdgroups x 4 rows per threadgroup.
template <int NT>
[[kernel]] void qmvt_t(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                       device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],
                       constant uint& K [[buffer(4)]], constant uint& O [[buffer(5)]], constant uint& add [[buffer(6)]],
                       constant uint& T [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                       uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * 2 + sid) * 4;
    uint kw = K / 16, kg = K / 128;
    float acc[NT][4];
    for (int t = 0; t < NT; t++) for (int r = 0; r < 4; r++) acc[t][r] = 0;
    for (uint k0 = lane * 16; k0 < K; k0 += 512) {
        float2 xv[NT][8];
        float xs[NT];
        for (int t = 0; t < NT; t++) {
            device const float* xp = X + min(uint(t), T - 1) * K + k0;
            float4 a0 = *(device const float4*)xp, a1 = *(device const float4*)(xp + 4);
            float4 b0 = *(device const float4*)(xp + 8), b1 = *(device const float4*)(xp + 12);
            float lo[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float hi[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            xs[t] = 0;
            for (uint j = 0; j < 8; j++) {
                float sc = j < 5 ? exp2(-2.0f * float(j)) : exp2(-2.0f * float(j - 5));
                xs[t] += lo[j] + hi[j];
                xv[t][j] = float2(lo[j], hi[j]) * sc;
            }
        }
        for (int r = 0; r < 4; r++) {
            uint wd = W[(row0 + r) * kw + k0 / 16];
            uint lo2 = wd, hi2 = wd >> 10;
            float2 f[8];
            for (uint j = 0; j < 8; j++) {
                uint m = j < 5 ? (lo2 & (0x00030003u << (2 * j))) : (hi2 & (0x00030003u << (2 * (j - 5))));
                f[j] = float2(as_type<half2>(m | 0x64006400u) - half2(1024.0h));
            }
            float sc = float(S[(row0 + r) * kg + k0 / 128]);
            for (int t = 0; t < NT; t++) {
                float d = 0;
                for (uint j = 0; j < 8; j++) d += xv[t][j].x * f[j].x + xv[t][j].y * f[j].y;
                acc[t][r] += sc * (d - xs[t]);
            }
        }
    }
    for (int t = 0; t < NT; t++) {
        for (int r = 0; r < 4; r++) {
            float v = simd_sum(acc[t][r]);
            if (lane == 0 && uint(t) < T) {
                uint o = t * O + row0 + r;
                Y[o] = add ? Y[o] + v : v;
            }
        }
    }
}
template [[host_name("qmvt2")]] [[kernel]] void qmvt_t<2>(device const uint*, device const half*, device const float*,
    device float*, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint);

// Up to three matvecs sharing one input in a single dispatch (rows of matrix i follow matrix i-1).
// Same inner loop as qmv4: 2 simdgroups x 4 rows per threadgroup, 16 codes per lane per step.
// LoRA epilogue when lr > 0: matrix i uses B_i and u + i * lr.
kernel void qmv3(device const uint* W0 [[buffer(0)]], device const half* S0 [[buffer(1)]], device float* Y0 [[buffer(2)]],
                 device const uint* W1 [[buffer(3)]], device const half* S1 [[buffer(4)]], device float* Y1 [[buffer(5)]],
                 device const uint* W2 [[buffer(6)]], device const half* S2 [[buffer(7)]], device float* Y2 [[buffer(8)]],
                 device const float* x [[buffer(9)]], constant uint& K [[buffer(10)]], constant uint& O0 [[buffer(11)]],
                 constant uint& O1 [[buffer(12)]], device const half* B0 [[buffer(13)]],
                 device const half* B1 [[buffer(14)]], device const half* B2 [[buffer(15)]],
                 device const float* u [[buffer(16)]], constant uint& lr [[buffer(17)]],
                 uint tg [[threadgroup_position_in_grid]],
                 uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = (tg * 2 + sid) * 4;
    device const uint* W; device const half* S; device float* y; device const half* B;
    if (r < O0) { W = W0; S = S0; y = Y0; B = B0; }
    else if (r < O0 + O1) { W = W1; S = S1; y = Y1; B = B1; r -= O0; u += lr; }
    else { W = W2; S = S2; y = Y2; B = B2; r -= O0 + O1; u += 2 * lr; }
    qmv_rows<4>(W, S, x, y, K, r, 0u, lane, B, u, lr);
}

// Prefill matmul: Y[T, O] (+)= X[T, K] W[O, K]^T on simdgroup MMA, 32 tokens x 64 rows per threadgroup, K step 32,
// 4 simdgroups each 32 x 16. The weight tile is exact fp16 ((code - 1) * scale is exact in half) and one 8 KB
// threadgroup buffer holds both tiles and the output staging, for occupancy. Grid (O / 64, ceil(T / 32)).
kernel void qmm3(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                 device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],
                 constant uint& K [[buffer(4)]], constant uint& O [[buffer(5)]], constant uint& add [[buffer(6)]],
                 constant uint& T [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]], uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[2048];
    threadgroup float* xs = buf;                              // [32 tokens][32 k] float
    threadgroup half* ws = (threadgroup half*)(buf + 1024);   // [64 rows][32 k] half
    uint n0 = tg.x * 64, t0 = tg.y * 32;
    uint kw = K / 16, kg = K / 128;
    simdgroup_float8x8 c[4][2];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint xr = tid / 4, xc = (tid % 4) * 8;
    uint wr = tid / 2, wq = tid % 2;
    for (uint k0 = 0; k0 < K; k0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tt = t0 + xr;
        float4 a = 0, b = 0;
        if (tt < T) { a = *(device const float4*)(X + tt * K + k0 + xc); b = *(device const float4*)(X + tt * K + k0 + xc + 4); }
        *(threadgroup float4*)(xs + xr * 32 + xc) = a;
        *(threadgroup float4*)(xs + xr * 32 + xc + 4) = b;
        uint row = n0 + wr;
        uint wd = W[row * kw + k0 / 16 + wq];
        half sc = S[row * kg + k0 / 128];
        threadgroup half* wp = ws + wr * 32 + wq * 16;
        for (uint j = 0; j < 8; j++) {
            half2 h = (as_type<half2>(((wd >> (2 * j)) & 0x00030003u) | 0x64006400u) - half2(1025.0h)) * sc;
            wp[j] = h.x;
            wp[8 + j] = h.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 am[4], bm[2];
            for (int i = 0; i < 4; i++) simdgroup_load(am[i], xs + i * 8 * 32 + kk, 32);
            for (int j = 0; j < 2; j++) {
                simdgroup_half8x8 bh;
                simdgroup_load(bh, ws + (sid * 16 + j * 8) * 32 + kk, 32, ulong2(0, 0), true);
                bm[j].thread_elements()[0] = float(bh.thread_elements()[0]);
                bm[j].thread_elements()[1] = float(bh.thread_elements()[1]);
            }
            for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) simdgroup_multiply_accumulate(c[i][j], am[i], bm[j], c[i][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) simdgroup_store(c[i][j], buf + i * 8 * 64 + sid * 16 + j * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * 64; e += 128) {
        uint tt = t0 + e / 64, n = n0 + e % 64;
        if (tt < T) {
            uint o = tt * O + n;
            float v = buf[e];
            Y[o] = add ? Y[o] + v : v;
        }
    }
}


// Dense fp32 matvec for the GDN a/b projections: out[t, 0..NV) = Wa x, out[t, NV..2NV) = Wb x. One simdgroup per
// row, grid (2 NV / 4, T).
kernel void gdn_ab(device const float* Wa [[buffer(0)]], device const float* Wb [[buffer(1)]],
                   device const float* x [[buffer(2)]], device float* out [[buffer(3)]],
                   constant uint& NV [[buffer(4)]], constant uint& K [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint sid [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg.x * 4 + sid;
    uint t = tg.y;
    device const float* w = row < NV ? Wa + row * K : Wb + (row - NV) * K;
    float acc = 0;
    for (uint i = lane * 4; i < K; i += 128) acc += dot(*(device const float4*)(w + i), *(device const float4*)(x + t * K + i));
    acc = simd_sum(acc);
    if (lane == 0) out[t * 2 * NV + row] = acc;
}

// Causal depthwise conv (kernel 4) over [state(3 rows), qkv(T rows)], silu, then q/k rmsnorm per head.
// C = conv channels (2 NK + NV heads of 128). Grid (T, C / 128), 128 threads. Output qkvc [T, C]:
// q * 1/128, k * 1/sqrt(128), v.
kernel void gdn_conv(device const float* qkv [[buffer(0)]], device const float* cstate [[buffer(1)]],
                     device const float* cw [[buffer(2)]], device float* out [[buffer(3)]],
                     constant uint& update [[buffer(4)]], constant uint& C [[buffer(5)]],
                     constant uint& NK [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint t = tg.x, grp = tg.y;
    uint c = grp * 128 + tid;
    float acc = 0;
    float rows[4];
    for (uint i = 0; i < 4; i++) {
        uint j = t + i;  // row in [state, qkv]
        rows[i] = j < 3 ? cstate[j * C + c] : qkv[(j - 3) * C + c];
        acc += cw[c * 4 + i] * rows[i];
    }
    if (update) {  // single-token decode: shift this channel's conv state in place
        device float* cs = (device float*)cstate;
        for (uint i = 0; i < 3; i++) cs[i * C + c] = rows[i + 1];
    }
    float v = acc / (1.0f + exp(-acc));
    if (grp < 2 * NK) {
        float r = rsqrt(tg_sum(v * v, red, sid, lane, 4) / 128.0f + EPS);
        v = v * r * (grp < NK ? (1.0f / 128.0f) : 0.08838834764831845f);
    }
    out[t * C + c] = v;
}

// New conv state = rows n, n+1, n+2 of [state_in(3), qkv]. One thread per channel; in place safe.
kernel void conv_state(device const float* qkv [[buffer(0)]], device const float* sin_ [[buffer(1)]],
                       device float* sout [[buffer(2)]], constant uint& n [[buffer(3)]],
                       constant uint& C [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    float r[3];
    for (uint j = 0; j < 3; j++) {
        uint k = n + j;
        r[j] = k < 3 ? sin_[k * C + c] : qkv[(k - 3) * C + c];
    }
    for (uint j = 0; j < 3; j++) sout[j * C + c] = r[j];
}

// Gated delta rule, sequential over T. State [NV, 128 (v), 128 (k)] fp32. V head h reads q/k head h / (NV / NK)
// (mlx-vlm's grouped layout). Grid (NV, 32 dv-groups), threadgroup (32 lanes over k, 4 dv rows). Each thread holds
// 4 k entries of one dv row. In place safe.
kernel void gdn_recurrent(device const float* qkvc [[buffer(0)]], device const float* ab [[buffer(1)]],
                          device const float* A_log [[buffer(2)]], device const float* dt_bias [[buffer(3)]],
                          device const float* sin_ [[buffer(4)]], device float* sout [[buffer(5)]],
                          device float* y [[buffer(6)]], constant uint& T [[buffer(7)]],
                          constant uint& NK [[buffer(8)]], constant uint& NV [[buffer(9)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]]) {
    uint hd = tg.x, lane = tp.x;
    uint hk = hd / (NV / NK);
    uint C = (2 * NK + NV) * 128;
    uint dv = tg.y * 4 + tp.y;
    uint base = (hd * 128 + dv) * 128 + lane * 4;
    float4 st = *(device const float4*)(sin_ + base);
    float ea = exp(A_log[hd]);
    float dtb = dt_bias[hd];
    for (uint t = 0; t < T; t++) {
        device const float* row = qkvc + t * C;
        float4 q = *(device const float4*)(row + hk * 128 + lane * 4);
        float4 k = *(device const float4*)(row + (NK + hk) * 128 + lane * 4);
        float v = row[(2 * NK + hd) * 128 + dv];
        float a = ab[t * 2 * NV + hd] + dtb;
        float sp = max(a, 0.0f) + log(1.0f + exp(-abs(a)));
        float g = exp(-ea * sp);
        float beta = 1.0f / (1.0f + exp(-ab[t * 2 * NV + NV + hd]));
        st *= g;
        float kv = simd_sum(dot(st, k));
        float delta = (v - kv) * beta;
        st += k * delta;
        float o = simd_sum(dot(st, q));
        if (lane == 0) y[t * NV * 128 + hd * 128 + dv] = o;
    }
    *(device float4*)(sout + base) = st;
}

// Attention prep: q/k rmsnorm (weights), interleaved mrope on the first ROT dims (sections s1, s2 for the h and w
// rows, the rest t), KV cache write. Grid (T, H + 2 KV): heads [0, H) q, then KV k heads, then KV v heads.
// 256 threads (head dim).
kernel void attn_prep(device const float* qp [[buffer(0)]], device const float* kp [[buffer(1)]],
                      device const float* vp [[buffer(2)]], device const float* qn [[buffer(3)]],
                      device const float* kn [[buffer(4)]], device const int* pos [[buffer(5)]],
                      device const float* inv_freq [[buffer(6)]], device float* qout [[buffer(7)]],
                      device float* kc [[buffer(8)]], device float* vc [[buffer(9)]],
                      constant uint& T [[buffer(10)]], constant uint& off [[buffer(11)]],
                      constant uint& H [[buffer(12)]], constant uint& KV [[buffer(13)]],
                      constant uint& ROT [[buffer(14)]], constant uint& s1 [[buffer(15)]],
                      constant uint& s2 [[buffer(16)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    threadgroup float buf[256];
    uint t = tg.x, hh = tg.y;
    if (hh >= H + KV) {
        uint kh = hh - H - KV;
        vc[((off + t) * KV + kh) * 256 + d] = vp[t * KV * 256 + kh * 256 + d];
        return;
    }
    bool isq = hh < H;
    float x = isq ? qp[t * H * 512 + hh * 512 + d] : kp[t * KV * 256 + (hh - H) * 256 + d];
    float r = rsqrt(tg_sum(x * x, red, sid, lane, 8) / 256.0f + EPS);
    x = x * r * (isq ? qn[d] : kn[d]);
    buf[d] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < ROT) {
        uint half_ = ROT / 2;
        uint i = d % half_;
        uint sel = (i % 3 == 1 && i < 3 * s1) ? 1u : ((i % 3 == 2 && i < 3 * s2) ? 2u : 0u);
        float ang = float(pos[sel * T + t]) * inv_freq[i];
        float cs = precise::cos(ang), sn = precise::sin(ang);
        x = d < half_ ? x * cs - buf[d + half_] * sn : x * cs + buf[d - half_] * sn;
    }
    if (isq) qout[(t * H + hh) * 256 + d] = x;
    else kc[((off + t) * KV + (hh - H)) * 256 + d] = x;
}
// Small-T matmul on MMA with X as the A operand (8 tokens x 8 k, plain device loads) and the weight tile as B
// (8 k x 8 rows), decoded in registers: lane owns (k = fm, rows fn and fn + 1). 4 simdgroups x 8 rows per
// threadgroup, K step 64, T <= 8.
kernel void qmvx(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                 device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],
                 constant uint& K [[buffer(4)]], constant uint& O [[buffer(5)]], constant uint& add [[buffer(6)]],
                 constant uint& T [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                 uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint n0 = (tg * 4 + sid) * 8;
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint kw = K / 16, kg = K / 128;
    device const uint* w0 = W + (n0 + fn) * kw;
    device const uint* w1 = w0 + kw;
    simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 64) {
        uint4 a4 = *(device const uint4*)(w0 + k0 / 16);
        uint4 b4 = *(device const uint4*)(w1 + k0 / 16);
        float s0 = float(S[(n0 + fn) * kg + k0 / 128]), s1 = float(S[(n0 + fn + 1) * kg + k0 / 128]);
        for (uint f = 0; f < 8; f++) {
            uint i = 8 * f + fm;  // code index within the 64-wide step
            uint sh = 2 * (i & 15);
            simdgroup_float8x8 a, b;
            simdgroup_load(a, X + k0 + 8 * f, K);
            b.thread_elements()[0] = (float((a4[i / 16] >> sh) & 3u) - 1.0f) * s0;
            b.thread_elements()[1] = (float((b4[i / 16] >> sh) & 3u) - 1.0f) * s1;
            simdgroup_multiply_accumulate(c, a, b, c);
        }
    }
    for (uint e = 0; e < 2; e++) {
        uint t = fm, n = n0 + fn + e;
        if (t < T) {
            uint o = t * O + n;
            float v = c.thread_elements()[e];
            Y[o] = add ? Y[o] + v : v;
        }
    }
}

// Prefill matmul, fp16 operands with fp32 accumulation: Y[T, O] (+)= X[T, K] W[O, K]^T. Tile BT tokens x 64 rows,
// K step 32, 4 simdgroups each 32 tokens x (64 / SO) rows (SO = 4 / (BT / 32)). X is rounded to half on load.
// Threadgroup tiles hold 8x8 blocks contiguously (X block [t][k], W block [k][o]) so every simdgroup_load reads
// 64 consecutive halves. Global loads for the next K step are issued before the MMAs of the current one.
template <int BT>
[[kernel]] void qmmh_t(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                       device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],
                       constant uint& K [[buffer(4)]], constant uint& O [[buffer(5)]], constant uint& add [[buffer(6)]],
                       constant uint& T [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr int ST = BT / 32, SO = 4 / ST, NB = 8 / SO;  // simdgroup grid, row blocks per simdgroup
    constexpr int XF = BT / 4;                             // X floats per thread per K step
    threadgroup half xs[BT * 32];
    threadgroup half ws[64 * 32];
    uint n0 = tg.x * 64, t0 = tg.y * BT;
    uint kw = K / 16, kg = K / 128;
    // X load: token xt, k offset xk (XF consecutive k)
    uint xt = tid / (32 / XF), xk = (tid % (32 / XF)) * XF;
    bool xin = t0 + xt < T;
    device const float* xp = X + min(t0 + xt, T - 1) * K + xk;
    // W decode: row pair (2 rp, 2 rp + 1), k block wk (8 codes)
    uint rp = tid % 32, wk = tid / 32;
    device const uint* w0 = W + (n0 + 2 * rp) * kw + wk / 2;
    device const half* s0 = S + (n0 + 2 * rp) * kg;
    uint sh0 = (wk % 2) * 16;
    float4 xr[XF / 4];
    uint wa, wb;
    half2 sc;
#define QMMH_LOAD(k0) \
    for (int i = 0; i < XF / 4; i++) xr[i] = xin ? *(device const float4*)(xp + (k0) + 4 * i) : float4(0); \
    wa = w0[(k0) / 16]; \
    wb = w0[kw + (k0) / 16]; \
    sc = half2(s0[(k0) / 128], s0[kg + (k0) / 128]);
    uint sto = sid / SO, soo = sid % SO;  // this simdgroup's token half and row slice
    simdgroup_float8x8 c[4][NB];
    for (int i = 0; i < 4; i++) for (int j = 0; j < NB; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    QMMH_LOAD(0)
    for (uint k0 = 0; k0 < K; k0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < XF / 4; i++) {
            uint k = xk + 4 * i;
            *(threadgroup half4*)(xs + ((xt / 8) * 4 + k / 8) * 64 + (xt % 8) * 8 + k % 8) = half4(xr[i]);
        }
        threadgroup half* wp = ws + ((2 * rp / 8) * 4 + wk) * 64 + (2 * rp) % 8;
        for (uint j = 0; j < 8; j++) {
            uint s = sh0 + 2 * j;
            uint m = ((wa >> s) & 3u) | (((wb >> s) & 3u) << 16);
            *(threadgroup half2*)(wp + j * 8) = (as_type<half2>(m | 0x64006400u) - half2(1025.0h)) * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 32 < K) { QMMH_LOAD(k0 + 32) }
        for (uint kb = 0; kb < 4; kb++) {
            simdgroup_half8x8 a[4], b[NB];
            for (int i = 0; i < 4; i++) simdgroup_load(a[i], xs + ((sto * 4 + i) * 4 + kb) * 64, 8);
            for (int j = 0; j < NB; j++) simdgroup_load(b[j], ws + ((soo * NB + j) * 4 + kb) * 64, 8);
            for (int i = 0; i < 4; i++) for (int j = 0; j < NB; j++) simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
        }
    }
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int i = 0; i < 4; i++) {
        uint t = t0 + (sto * 4 + i) * 8 + fm;
        if (t >= T) continue;
        for (int j = 0; j < NB; j++) {
            device float2* yp = (device float2*)(Y + t * O + n0 + (soo * NB + j) * 8 + fn);
            float2 v = float2(c[i][j].thread_elements()[0], c[i][j].thread_elements()[1]);
            *yp = add ? *yp + v : v;
        }
    }
}
template [[host_name("qmmh32")]] [[kernel]] void qmmh_t<32>(device const uint*, device const half*, device const float*,
    device float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint);
template [[host_name("qmmh64")]] [[kernel]] void qmmh_t<64>(device const uint*, device const half*, device const float*,
    device float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint);

// Small-T matmul (verify, T <= 8 NA) on fp16 MMA: X rows as NA A fragments (rounded to half), the weight fragment
// decoded in registers as B and applied to all of them, so each weight is read once for all T rows. Lane owns
// B (k = fm, rows fn, fn + 1). Threadgroup = 4 simdgroups: 4 / NKS row groups of 8 rows, NKS simdgroups splitting
// K per group, partials summed in threadgroup memory. K step 64.
template <int NA, int NKS>
[[kernel]] void qmvh_t(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                       device const float* X [[buffer(2)]], device float* Y [[buffer(3)]],
                       constant uint& K [[buffer(4)]], constant uint& O [[buffer(5)]], constant uint& add [[buffer(6)]],
                       constant uint& T [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                       uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[NKS > 1 ? (4 / NKS) * (NKS - 1) * NA * 64 : 1];
    uint rg = sid / NKS, ks = sid % NKS;
    uint n0 = (tg * (4 / NKS) + rg) * 8;
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint kw = K / 16, kg = K / 128;
    device const uint* w0 = W + (n0 + fn) * kw;
    device const uint* w1 = w0 + kw;
    device const half* sp = S + (n0 + fn) * kg;
    simdgroup_float8x8 c[NA];
    for (int a = 0; a < NA; a++) c[a] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = ks * 64; k0 < K; k0 += 64 * NKS) {
        uint4 a4 = *(device const uint4*)(w0 + k0 / 16);
        uint4 b4 = *(device const uint4*)(w1 + k0 / 16);
        half2 sc = half2(sp[k0 / 128], sp[kg + k0 / 128]);
        for (uint f = 0; f < 8; f++) {
            uint i = 8 * f + fm;
            uint sh = 2 * (i & 15);
            uint m = ((a4[i / 16] >> sh) & 3u) | (((b4[i / 16] >> sh) & 3u) << 16);
            half2 h = (as_type<half2>(m | 0x64006400u) - half2(1025.0h)) * sc;
            simdgroup_half8x8 b;
            b.thread_elements()[0] = h.x;
            b.thread_elements()[1] = h.y;
            for (int a = 0; a < NA; a++) {
                simdgroup_float8x8 xf;
                simdgroup_load(xf, X + a * 8 * K + k0 + 8 * f, K);
                simdgroup_half8x8 xh;
                xh.thread_elements()[0] = half(xf.thread_elements()[0]);
                xh.thread_elements()[1] = half(xf.thread_elements()[1]);
                simdgroup_multiply_accumulate(c[a], xh, b, c[a]);
            }
        }
    }
    if (NKS > 1) {
        threadgroup float* r = red + (rg * (NKS - 1) * NA) * 64;
        if (ks > 0) for (int a = 0; a < NA; a++) simdgroup_store(c[a], r + ((ks - 1) * NA + a) * 64, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks > 0) return;
        for (uint s = 1; s < NKS; s++) {
            for (int a = 0; a < NA; a++) {
                simdgroup_float8x8 p;
                simdgroup_load(p, r + ((s - 1) * NA + a) * 64, 8);
                c[a].thread_elements()[0] += p.thread_elements()[0];
                c[a].thread_elements()[1] += p.thread_elements()[1];
            }
        }
    }
    for (int a = 0; a < NA; a++) {
        uint t = a * 8 + fm;
        if (t >= T) continue;
        device float2* yp = (device float2*)(Y + t * O + n0 + fn);
        float2 v = float2(c[a].thread_elements()[0], c[a].thread_elements()[1]);
        *yp = add ? *yp + v : v;
    }
}
#define QMVH(NA, NKS) template [[host_name("qmvh" #NA "_" #NKS)]] [[kernel]] void qmvh_t<NA, NKS>(device const uint*, \
    device const half*, device const float*, device float*, constant uint&, constant uint&, constant uint&, \
    constant uint&, uint, uint, uint);
QMVH(1, 1) QMVH(1, 2) QMVH(1, 4) QMVH(2, 1) QMVH(2, 2) QMVH(2, 4)

// Split-K causal attention. Threadgroup = (kv head, key chunk, query t); the 4 q heads of that kv head share
// each K/V read. Writes per-chunk partials (max, sum, acc[256]) to part[(t * H + h) * nsplit + s]. 256 threads.
kernel void attn_split(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]],
                       device const float* vc [[buffer(2)]], device float* part [[buffer(3)]],
                       constant uint& off [[buffer(4)]], constant uint& chunk [[buffer(5)]],
                       constant uint& nsplit [[buffer(6)]], constant uint& H [[buffer(7)]],
                       constant uint& KV [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sc[4 * 512];
    threadgroup float ml[8];
    uint kh = tg.x, s = tg.y, t = tg.z;
    uint n = off + t + 1;
    uint j0 = s * chunk, j1 = min(n, j0 + chunk);
    device float* pp = part + ((t * H + kh * 4) * nsplit + s) * 258;
    if (j0 >= j1) {
        if (tid < 4) { pp[tid * nsplit * 258] = -INFINITY; pp[tid * nsplit * 258 + 1] = 0; }
        return;
    }
    float4 qa[4], qb[4];
    for (uint h = 0; h < 4; h++) {
        device const float* qp = q + (t * H + kh * 4 + h) * 256 + lane * 8;
        qa[h] = *(device const float4*)qp * 0.0625f;
        qb[h] = *(device const float4*)(qp + 4) * 0.0625f;
    }
    for (uint j = j0 + sid; j < j1; j += 8) {
        device const float* k = kc + (j * KV + kh) * 256 + lane * 8;
        float4 ka = *(device const float4*)k, kb = *(device const float4*)(k + 4);
        for (uint h = 0; h < 4; h++) {
            float d = simd_sum(dot(qa[h], ka) + dot(qb[h], kb));
            if (lane == 0) sc[h * 512 + (j - j0)] = d;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cnt = j1 - j0;
    if (sid < 4) {
        float m = -INFINITY;
        for (uint i = lane; i < cnt; i += 32) m = max(m, sc[sid * 512 + i]);
        m = simd_max(m);
        float l = 0;
        for (uint i = lane; i < cnt; i += 32) { float p = exp(sc[sid * 512 + i] - m); sc[sid * 512 + i] = p; l += p; }
        l = simd_sum(l);
        if (lane == 0) { ml[sid * 2] = m; ml[sid * 2 + 1] = l; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[4] = {0, 0, 0, 0};
    for (uint i = 0; i < cnt; i++) {
        float v = vc[((j0 + i) * KV + kh) * 256 + tid];
        for (uint h = 0; h < 4; h++) acc[h] += sc[h * 512 + i] * v;
    }
    for (uint h = 0; h < 4; h++) {
        device float* ph = pp + h * nsplit * 258;
        if (tid == 0) { ph[0] = ml[h * 2]; ph[1] = ml[h * 2 + 1]; }
        ph[2 + tid] = acc[h];
    }
}

// Combine split partials: out[t, h * 256 + d]. Grid (H, T), 256 threads.
kernel void attn_combine(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& nsplit [[buffer(2)]], constant uint& H [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    uint h = tg.x, t = tg.y;
    device const float* p = part + (t * H + h) * nsplit * 258;
    float M = -INFINITY;
    for (uint s = 0; s < nsplit; s++) M = max(M, p[s * 258]);
    float L = 0, a = 0;
    for (uint s = 0; s < nsplit; s++) {
        float m = p[s * 258];
        if (m == -INFINITY) continue;
        float w = exp(m - M);
        L += p[s * 258 + 1] * w;
        a += p[s * 258 + 2 + d] * w;
    }
    out[t * H * 256 + h * 256 + d] = a / L;
}

// LoRA up projection for T > 1 (decode uses the qmv epilogue), added to up to three outputs:
// Y_i[t, o] += B_i[o] . u[t, i r .. (i + 1) r), B_i fp16 [O_i, r] (alpha / r folded in). Rows of output i follow output i - 1. Grid (ceil(total / 256), T), 256 threads.
kernel void lora_b(device const half* B0 [[buffer(0)]], device float* Y0 [[buffer(1)]],
                   device const half* B1 [[buffer(2)]], device float* Y1 [[buffer(3)]],
                   device const half* B2 [[buffer(4)]], device float* Y2 [[buffer(5)]],
                   device const float* u [[buffer(6)]], constant uint& O0 [[buffer(7)]], constant uint& O1 [[buffer(8)]],
                   constant uint& O2 [[buffer(9)]], constant uint& r [[buffer(10)]], constant uint& R [[buffer(11)]],
                   uint2 gp [[thread_position_in_grid]]) {
    uint o = gp.x, t = gp.y;
    device const half* b; device float* y; uint O, i;
    if (o < O0) { b = B0; y = Y0; O = O0; i = 0; }
    else if (o < O0 + O1) { b = B1; y = Y1; O = O1; i = 1; o -= O0; }
    else if (o < O0 + O1 + O2) { b = B2; y = Y2; O = O2; i = 2; o -= O0 + O1; }
    else return;
    device const float* ut = u + t * R + i * r;
    device const half* br = b + o * r;
    float acc = 0;
    for (uint j = 0; j < r; j++) acc += ut[j] * float(br[j]);
    y[t * O + o] += acc;
}

// Argmax over V per row, stage 1: each threadgroup scans a slice, writes (value, index).
kernel void argmax1(device const float* x [[buffer(0)]], device float* pv [[buffer(1)]], device int* pi [[buffer(2)]],
                    constant uint& V [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint2 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float rv[32];
    threadgroup int ri[32];
    uint row = tg.y;
    uint chunk = (V + ng.x - 1) / ng.x;
    uint b = tg.x * chunk, e = min(V, b + chunk);
    float bv = -INFINITY;
    int bi = int(b);
    for (uint i = b + tid; i < e; i += 256) {
        float v = x[row * V + i];
        if (v > bv) { bv = v; bi = int(i); }
    }
    for (uint o = 16; o > 0; o >>= 1) {
        float ov = simd_shuffle_down(bv, o);
        int oi = simd_shuffle_down(bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { rv[sid] = bv; ri[sid] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < 8; s++) if (rv[s] > bv || (rv[s] == bv && ri[s] < bi)) { bv = rv[s]; bi = ri[s]; }
        pv[row * ng.x + tg.x] = bv;
        pi[row * ng.x + tg.x] = bi;
    }
}

kernel void argmax2(device const float* pv [[buffer(0)]], device const int* pi [[buffer(1)]],
                    device int* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint row [[thread_position_in_grid]]) {
    float bv = -INFINITY;
    int bi = 0;
    for (uint i = 0; i < n; i++) {
        float v = pv[row * n + i];
        int ix = pi[row * n + i];
        if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
    }
    out[row] = bi;
}

// ---------------- Vision tower (fp16 activations and weights, fp32 accumulation, like mlx-vlm in fp16) --------

inline float gelu_tanh(float x) { return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
// erf via Numerical Recipes erfc Chebyshev fit (relative error < 1.2e-7).
inline float erf_(float x) {
    float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
    float r = t * exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
              t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0 ? 1.0f - r : r - 1.0f;
}
inline float gelu_erf(float x) { return 0.5f * x * (1.0f + erf_(x * 0.7071067811865476f)); }

// Y[M, N] = act(X[M, K] W[N, K]^T + bias) (+ R). Half in/out. Tile 32 x 64, K step 32, 4 simdgroups.
// act: 0 none, 1 gelu tanh, 2 gelu erf. Rounds to half after bias, after act and after the residual add.
kernel void v_gemm(device const half* X [[buffer(0)]], device const half* W [[buffer(1)]],
                   device const half* bias [[buffer(2)]], device half* Y [[buffer(3)]],
                   constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& K [[buffer(6)]],
                   constant uint& act [[buffer(7)]], constant uint& add [[buffer(8)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup half xs[32 * 32];
    threadgroup half ws[64 * 32];
    threadgroup float out[32 * 64];
    uint n0 = tg.x * 64, m0 = tg.y * 32;
    simdgroup_float8x8 c[4][2];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint xr = tid / 4, xc = (tid % 4) * 8;
    uint wr = tid / 2, wc = (tid % 2) * 16;
    for (uint k0 = 0; k0 < K; k0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint mm = m0 + xr;
        half4 a = 0, b = 0;
        if (mm < M) { a = *(device const half4*)(X + mm * K + k0 + xc); b = *(device const half4*)(X + mm * K + k0 + xc + 4); }
        *(threadgroup half4*)(xs + xr * 32 + xc) = a;
        *(threadgroup half4*)(xs + xr * 32 + xc + 4) = b;
        device const half* wp = W + (n0 + wr) * K + k0 + wc;
        for (uint j = 0; j < 16; j += 4) *(threadgroup half4*)(ws + wr * 32 + wc + j) = *(device const half4*)(wp + j);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 am[4], bm[2];
            for (int i = 0; i < 4; i++) simdgroup_load(am[i], xs + i * 8 * 32 + kk, 32);
            for (int j = 0; j < 2; j++) simdgroup_load(bm[j], ws + (sid * 16 + j * 8) * 32 + kk, 32, ulong2(0, 0), true);
            for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) simdgroup_multiply_accumulate(c[i][j], am[i], bm[j], c[i][j]);
        }
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) simdgroup_store(c[i][j], out + i * 8 * 64 + sid * 16 + j * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * 64; e += 128) {
        uint mm = m0 + e / 64, n = n0 + e % 64;
        if (mm < M) {
            float v = float(half(out[e] + float(bias[n])));
            if (act == 1) v = float(half(gelu_tanh(v)));
            else if (act == 2) v = float(half(gelu_erf(v)));
            uint o = mm * N + n;
            if (add) v = float(Y[o]) + v;
            Y[o] = half(v);
        }
    }
}

// LayerNorm over D (fp32 statistics), eps 1e-6, half in/out. One threadgroup (256) per row.
kernel void v_layernorm(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                        device const half* b [[buffer(2)]], device half* y [[buffer(3)]], constant uint& D [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    device const half* xr = x + row * D;
    float s = 0;
    for (uint i = tid; i < D; i += 256) s += float(xr[i]);
    float mean = tg_sum(s, red, sid, lane, 8) / float(D);
    float v = 0;
    for (uint i = tid; i < D; i += 256) { float d = float(xr[i]) - mean; v += d * d; }
    float r = rsqrt(tg_sum(v, red, sid, lane, 8) / float(D) + 1e-6f);
    for (uint i = tid; i < D; i += 256) y[row * D + i] = half((float(xr[i]) - mean) * r * float(w[i]) + float(b[i]));
}

// qkv [N, 3, 16, 64] -> q, k (rotary applied), v as [16][N][64]. cs: per patch 32 angles (16 row, 16 col).
// grid (N, 3 * 16), 64 threads.
kernel void v_rope(device const half* qkv [[buffer(0)]], device const float* ang [[buffer(1)]],
                   device half* q [[buffer(2)]], device half* k [[buffer(3)]], device half* v [[buffer(4)]],
                   constant uint& N [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    uint p = tg.x, which = tg.y / 16, hd = tg.y % 16;
    device const half* src = qkv + p * 3072 + which * 1024 + hd * 64;
    uint o = (hd * N + p) * 64 + d;
    if (which == 2) { v[o] = src[d]; return; }
    float a = ang[p * 32 + (d & 31)];
    float x = float(src[d]);
    float rot = d < 32 ? -float(src[d + 32]) : float(src[d - 32]);
    float r = x * precise::cos(a) + rot * precise::sin(a);
    (which == 0 ? q : k)[o] = half(r);
}

// Flash attention for the vision tower on simdgroup MMA. Each simdgroup owns 8 queries of one head and walks the
// keys in blocks of 8 with an online softmax; P is rounded to half for P V like MLX's fp16 attention.
// q, k, v: [16][N][64] half. Grid (ceil(N / 32), 16), 128 threads (4 simdgroups x 8 queries).
kernel void v_flash(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
                    device half* out [[buffer(3)]], constant uint& N [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    uint hd = tg.y;
    uint q0 = (tg.x * 4 + sid) * 8;
    if (q0 >= N) return;
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    device const half* qh = q + hd * N * 64;
    device const half* kh = k + hd * N * 64;
    device const half* vh = v + hd * N * 64;
    // Q fragments (8 queries x 64 dims); rows past N are clamped to N - 1 and never stored.
    simdgroup_half8x8 qf[8];
    uint qrow = min(q0 + fm, N - 1);
    for (uint f = 0; f < 8; f++) {
        qf[f].thread_elements()[0] = qh[qrow * 64 + f * 8 + fn];
        qf[f].thread_elements()[1] = qh[qrow * 64 + f * 8 + fn + 1];
    }
    simdgroup_float8x8 o[8];
    for (uint f = 0; f < 8; f++) o[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0;
    for (uint j0 = 0; j0 < N; j0 += 8) {
        simdgroup_float8x8 sfr = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint f = 0; f < 8; f++) {
            simdgroup_half8x8 kt;
            simdgroup_load(kt, kh + j0 * 64 + f * 8, 64, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sfr, qf[f], kt, sfr);
        }
        float s0 = sfr.thread_elements()[0] * 0.125f, s1 = sfr.thread_elements()[1] * 0.125f;
        if (j0 + fn >= N) s0 = -INFINITY;
        if (j0 + fn + 1 >= N) s1 = -INFINITY;
        float mx = max(s0, s1);
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        float mn = max(m, mx);
        float c = exp(m - mn);
        float p0 = exp(s0 - mn), p1 = exp(s1 - mn);
        float ps = p0 + p1;
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 8);
        l = l * c + ps;
        m = mn;
        simdgroup_half8x8 pf;
        pf.thread_elements()[0] = half(p0);
        pf.thread_elements()[1] = half(p1);
        for (uint f = 0; f < 8; f++) {
            o[f].thread_elements()[0] *= c;
            o[f].thread_elements()[1] *= c;
            simdgroup_half8x8 vf;
            simdgroup_load(vf, vh + j0 * 64 + f * 8, 64);
            simdgroup_multiply_accumulate(o[f], pf, vf, o[f]);
        }
    }
    uint row = q0 + fm;
    if (row < N) {
        for (uint f = 0; f < 8; f++) {
            out[row * 1024 + hd * 64 + f * 8 + fn] = half(o[f].thread_elements()[0] / l);
            out[row * 1024 + hd * 64 + f * 8 + fn + 1] = half(o[f].thread_elements()[1] / l);
        }
    }
}
