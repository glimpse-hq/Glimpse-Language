// Rotation kernel helpers: need `shared float red[16]; shared float lred[16];` and a push block `pc` with A, U, R.

// Sum of squares of row x[base .. base + D) by a 128-thread workgroup, as kernels.metal row_ss.
float row_ss(PTR x, uint base, uint D) {
    float ss[4] = float[4](0.0, 0.0, 0.0, 0.0);
    for (uint i0 = 0u; i0 < D; i0 += 512u)
        [[unroll]] for (uint q = 0u; q < 4u; q++) {
            uint i = i0 + (SID * 4u + q) * 32u + LANE;
            if (i < D) { float v = Fp(x).v[base + i]; ss[q] += v * v; }
        }
    [[unroll]] for (uint q = 0u; q < 4u; q++) {
        float p = subgroupAdd(ss[q]);
        if (LANE == 0u) red[SID * 4u + q] = p;
    }
    barrier();
    return subgroupAdd(LANE < 16u ? red[LANE] : 0.0);
}

// Fused LoRA A: workgroups past the rotation chunks each compute rpt rows of U[t, row] = A[row] . EXPR(k).
#define LORA_RPT_MAX 4u
#define LORA_ROW(nch, EXPR)                                                                         \
    if (gl_WorkGroupID.y >= nch) {                                                                  \
        uint ny_ = gl_NumWorkGroups.y;                                                              \
        uint rpt = (pc.R + ny_ - nch - 1u) / (ny_ - nch), row0 = (gl_WorkGroupID.y - nch) * rpt;    \
        for (uint q = 0u; q < rpt; q++) {                                                           \
            uint row = row0 + q;                                                                    \
            float acc = 0.0;                                                                        \
            if (row < pc.R) {                                                                       \
                for (uint k = (SID * 32u + LANE) * 4u; k < K; k += 512u)                            \
                    acc += dot(half4_at(pc.A, row * K + k), (EXPR));                                \
            }                                                                                       \
            acc = subgroupAdd(acc);                                                                 \
            if (LANE == 0u) lred[q * 4u + SID] = acc;                                               \
        }                                                                                           \
        barrier();                                                                                  \
        uint qq = SID * 32u + LANE;                                                                 \
        if (qq < rpt && row0 + qq < pc.R)                                                           \
            Fp(pc.U).v[t * pc.R + row0 + qq] = lred[qq * 4u] + lred[qq * 4u + 1u] + lred[qq * 4u + 2u] + lred[qq * 4u + 3u]; \
        return;                                                                                     \
    }
