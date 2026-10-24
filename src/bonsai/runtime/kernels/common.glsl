// Shared definitions for the Vulkan kernels, ports of kernels.metal (same names, arguments and math).
// Arguments arrive as push constants in Metal buffer-index order: buffers as 64-bit device addresses (uvec2),
// scalars as uint. Subgroups are 32 lanes (required at pipeline creation), so lane / sid match Metal's simdgroups.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_control_flow_attributes : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Fp { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer F4p { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Up { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer U4p { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Ip { int v[]; };

#define PTR uvec2
#define LANE gl_SubgroupInvocationID
#define SID gl_SubgroupID

const float INV_SQRT512 = 0.044194173824159216;
const float EPS = 1e-6;
const uint CHUNK = 2048u;
const float NEG_INF = -1.0 / 0.0;

// Element i of a half array stored as 32-bit words.
float half_at(PTR p, uint i) {
    vec2 h = unpackHalf2x16(Up(p).v[i >> 1]);
    return (i & 1u) != 0u ? h.y : h.x;
}

// Four halves at element i (i a multiple of 4, array 8-byte aligned at i).
vec4 half4_at(PTR p, uint i) {
    Up u = Up(p);
    return vec4(unpackHalf2x16(u.v[i >> 1]), unpackHalf2x16(u.v[(i >> 1) + 1u]));
}

float round_half(float x) {
    return unpackHalf2x16(packHalf2x16(vec2(x, 0.0))).x;
}

vec4 f4(PTR p, uint i) {
    return F4p(p).v[i >> 2];
}

vec4 silu4(vec4 a) {
    return a / (1.0 + exp(-a));
}

vec4 sigmoid4(vec4 a) {
    return 1.0 / (1.0 + exp(-a));
}

// 512-point unnormalized Walsh-Hadamard transform of one block held by a subgroup: lane l holds elements
// l + 32 e (e = 0..15). Same butterflies and stage order as kernels.metal fwht512.
void fwht512(inout float v[16]) {
    [[unroll]] for (uint m = 1u; m < 32u; m <<= 1) {
        bool up = (LANE & m) != 0u;
        [[unroll]] for (uint e = 0u; e < 16u; e++) {
            float o = subgroupShuffleXor(v[e], m);
            v[e] = up ? o - v[e] : v[e] + o;
        }
    }
    [[unroll]] for (uint h = 1u; h < 16u; h <<= 1)
        [[unroll]] for (uint e = 0u; e < 16u; e++)
            if ((e & h) == 0u) {
                float a = v[e], b = v[e + h];
                v[e] = a + b;
                v[e + h] = a - b;
            }
}

// Cooperative-matrix tiles store k inside each group of 16 in the order 0, 8, 1, 9, ..., 7, 15, so one packed
// half pair holds codes e and e + 8 of a weight word (bits 2e and 16 + 2e): decoded with one shift and mask.
// Activation tiles use the same order, so the products are unchanged.
#ifdef CM_TILES
// 16 codes of word w, times scale, as 8 packed half pairs.
void decode_word(uint w, float16_t sc, out uint p[8]) {
    f16vec2 s2 = f16vec2(sc, sc);
    [[unroll]] for (uint e = 0u; e < 8u; e++) {
        f16vec2 c = unpackFloat2x16(((w >> (2u * e)) & 0x00030003u) | 0x64006400u);
        p[e] = packFloat2x16((c - f16vec2(1025.0)) * s2);
    }
}

// 16 activations (x[0..16) of one k group) as hi / lo packed half pairs in tile order.
void split16(vec4 a, vec4 b, vec4 c, vec4 d, out uvec4 h0, out uvec4 h1, out uvec4 l0, out uvec4 l1) {
    float x[16] = float[16](a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w);
    uint hp[8], lp[8];
    [[unroll]] for (uint e = 0u; e < 8u; e++) {
        vec2 v = vec2(x[e], x[e + 8u]);
        f16vec2 h = f16vec2(v);
        hp[e] = packFloat2x16(h);
        lp[e] = packFloat2x16(f16vec2(v - vec2(h)));
    }
    h0 = uvec4(hp[0], hp[1], hp[2], hp[3]); h1 = uvec4(hp[4], hp[5], hp[6], hp[7]);
    l0 = uvec4(lp[0], lp[1], lp[2], lp[3]); l1 = uvec4(lp[4], lp[5], lp[6], lp[7]);
}

// 16 fp16 values v[0..16) from a half array (LoRA B row), zero past n, as packed pairs in tile order.
void pack16_half(PTR b, uint base, uint j0, uint n, out uvec4 p0, out uvec4 p1) {
    uint p[8];
    [[unroll]] for (uint e = 0u; e < 8u; e++) {
        float a = j0 + e < n ? half_at(b, base + j0 + e) : 0.0;
        float c = j0 + e + 8u < n ? half_at(b, base + j0 + e + 8u) : 0.0;
        p[e] = packHalf2x16(vec2(a, c));
    }
    p0 = uvec4(p[0], p[1], p[2], p[3]); p1 = uvec4(p[4], p[5], p[6], p[7]);
}
#endif
