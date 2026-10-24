// Ternary matvec helpers. Packed rows: 16 2-bit codes per uint (code e at bits 2e), one fp16 scale per 128 codes;
// weight = scale * (code - 1).

float lora_acc(PTR u, uint uo, PTR b, uint bo, uint r) {
    float acc = 0.0;
    for (uint j = 0u; j < r; j++) acc += Fp(u).v[uo + j] * half_at(b, bo + j);
    return acc;
}

// y = (add ? y + v : v) + lora_acc, the order of a separate LoRA pass after the matmul.
void store_post(PTR y, uint yi, float v, uint add, PTR b, uint bo, PTR u, uint uo, uint r) {
    float yv = add != 0u ? Fp(y).v[yi] + v : v;
    if (r != 0u) yv = yv + lora_acc(u, uo, b, bo, r);
    Fp(y).v[yi] = yv;
}

// One subgroup computes 4 rows from row0 (16 codes per lane per step), plus the LoRA epilogue
// y[row] += B[row, 0..lr) . u when lr > 0. x at xo, y at yo.
void qmv_rows4(PTR W, PTR S, PTR x, uint xo, PTR y, uint yo, uint K, uint row0, uint add, PTR B, PTR u, uint uo, uint lr) {
    uint kw = K / 16u, kg = K / 128u;
    float acc[4] = float[4](0.0, 0.0, 0.0, 0.0);
    for (uint k0 = LANE * 16u; k0 < K; k0 += 512u) {
        vec4 a0 = f4(x, xo + k0), a1 = f4(x, xo + k0 + 4u), b0 = f4(x, xo + k0 + 8u), b1 = f4(x, xo + k0 + 12u);
        float lo[8] = float[8](a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w);
        float hi[8] = float[8](b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w);
        float xs = 0.0;
        [[unroll]] for (uint j = 0u; j < 8u; j++) xs += lo[j] + hi[j];
        [[unroll]] for (uint r = 0u; r < 4u; r++) {
            uint wd = Up(W).v[(row0 + r) * kw + k0 / 16u];
            float d = 0.0;
            [[unroll]] for (uint j = 0u; j < 8u; j++)
                d += lo[j] * float((wd >> (2u * j)) & 3u) + hi[j] * float((wd >> (16u + 2u * j)) & 3u);
            acc[r] += half_at(S, (row0 + r) * kg + k0 / 128u) * (d - xs);
        }
    }
    for (uint j = LANE; j < lr; j += 32u) {
        float uj = Fp(u).v[uo + j];
        [[unroll]] for (uint r = 0u; r < 4u; r++) acc[r] += uj * half_at(B, (row0 + r) * lr + j);
    }
    [[unroll]] for (uint r = 0u; r < 4u; r++) {
        float v = subgroupAdd(acc[r]);
        if (LANE == 0u) {
            uint yi = yo + row0 + r;
            Fp(y).v[yi] = add != 0u ? Fp(y).v[yi] + v : v;
        }
    }
}

// One GDN a/b gate row (dense fp32 matvec), one subgroup.
void gdn_ab_row(PTR Wa, PTR Wb, PTR x, PTR out_, uint NV, uint K, uint row, uint t) {
    PTR w = row < NV ? Wa : Wb;
    uint wo = (row < NV ? row : row - NV) * K;
    float acc = 0.0;
    for (uint i = LANE * 4u; i < K; i += 128u) acc += dot(f4(w, wo + i), f4(x, t * K + i));
    acc = subgroupAdd(acc);
    if (LANE == 0u) Fp(out_).v[t * 2u * NV + row] = acc;
}

#ifdef QMV_SPLIT
shared float qred[16];
// T = 1 matvec, 4 rows from row0 per workgroup of 4 subgroups: subgroup s takes k positions (s * 32 + lane) * 16
// + 2048 j, so each lane runs K / 2048 steps, and the 4 partial sums meet in shared memory. LoRA epilogue
// y[row] += B[row, 0..lr) . u[uo..] on subgroup 0.
void qmv_split4(PTR W, PTR S, PTR x, PTR y, uint K, uint row0, uint add, PTR B, PTR u, uint uo, uint lr) {
    uint kw = K / 16u, kg = K / 128u;
    float acc[4] = float[4](0.0, 0.0, 0.0, 0.0);
    for (uint k0 = (SID * 32u + LANE) * 16u; k0 < K; k0 += 2048u) {
        vec4 a0 = f4(x, k0), a1 = f4(x, k0 + 4u), b0 = f4(x, k0 + 8u), b1 = f4(x, k0 + 12u);
        float lo[8] = float[8](a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w);
        float hi[8] = float[8](b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w);
        float xs = 0.0;
        [[unroll]] for (uint j = 0u; j < 8u; j++) xs += lo[j] + hi[j];
        [[unroll]] for (uint r = 0u; r < 4u; r++) {
            uint wd = Up(W).v[(row0 + r) * kw + k0 / 16u];
            float d = 0.0;
            [[unroll]] for (uint j = 0u; j < 8u; j++)
                d += lo[j] * float((wd >> (2u * j)) & 3u) + hi[j] * float((wd >> (16u + 2u * j)) & 3u);
            acc[r] += half_at(S, (row0 + r) * kg + k0 / 128u) * (d - xs);
        }
    }
    if (SID == 0u)
        for (uint j = LANE; j < lr; j += 32u) {
            float uj = Fp(u).v[uo + j];
            [[unroll]] for (uint r = 0u; r < 4u; r++) acc[r] += uj * half_at(B, (row0 + r) * lr + j);
        }
    [[unroll]] for (uint r = 0u; r < 4u; r++) {
        float v = subgroupAdd(acc[r]);
        if (LANE == 0u) qred[SID * 4u + r] = v;
    }
    barrier();
    uint r = gl_LocalInvocationIndex;
    if (r < 4u) {
        float v = qred[r] + qred[4u + r] + qred[8u + r] + qred[12u + r];
        Fp(y).v[row0 + r] = add != 0u ? Fp(y).v[row0 + r] + v : v;
    }
}
#endif
