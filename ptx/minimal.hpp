// Minimal PTX paths: 12 common operations; sm_120 builds append EX2.BF16 and TANH.BF16.
// This layer models the PTX/SASS-visible surrounding instructions around the bare MUFU model.
// The f32 paths use .ftz except tanh (no .ftz form); f16, f64, and BF16 use their PTX forms.
#pragma once
#include "mufu_bits.hpp"
#include "mufu_model.h"
#ifdef SFU_ENABLE_BF16
#include "bf16.hpp"
#endif

namespace mufu_model { namespace minimal {

enum class Path : int {
    EX2, LG2, RCP, RSQ, SQRT, SIN, COS, TANH, RCP64H, RSQ64H, EX2_F16, TANH_F16,
#ifdef SFU_ENABLE_BF16
    EX2_BF16, TANH_BF16,
#endif
    Count
};

constexpr int kPathCount = static_cast<int>(Path::Count);
constexpr const char* kPathNames[kPathCount] = {
    "ex2.approx.ftz.f32", "lg2.approx.ftz.f32", "rcp.approx.ftz.f32", "rsqrt.approx.ftz.f32",
    "sqrt.approx.ftz.f32", "sin.approx.ftz.f32", "cos.approx.ftz.f32", "tanh.approx.f32",
    "rcp.approx.ftz.f64", "rsqrt.approx.ftz.f64", "ex2.approx.f16", "tanh.approx.f16",
#ifdef SFU_ENABLE_BF16
    "ex2.approx.ftz.bf16", "tanh.approx.bf16",
#endif
};

// Models the FMUL.RZ by 0x3e22f983 (1/(2*pi)) that precedes MUFU.SIN / MUFU.COS.
inline uint32_t prescale_rz(uint32_t bits) {
    const uint32_t sign = bits & kSignMask, magnitude = bits & ~kSignMask;
    if (magnitude >= kInfBits) return magnitude > kInfBits ? kNanBits : bits;
    if (!magnitude) return bits;
    const uint32_t exp_bits = (magnitude & kExpMask) >> 23;
    const uint32_t sig = exp_bits ? ((1u << 23) | (magnitude & kFracMask)) : (magnitude & kFracMask);
    const int exponent = exp_bits ? static_cast<int>(exp_bits) - 127 : -126;
    const uint64_t product = static_cast<uint64_t>(sig) * ((1u << 23) | (0x3e22f983u & kFracMask));
    const int power = exponent + (static_cast<int>((0x3e22f983u & kExpMask) >> 23) - 127) - 46;
    const int top = highest64(product);
    const int output_exp = top + power;
    if (output_exp >= -126) {
        const uint32_t significand = static_cast<uint32_t>(product >> (top - 23));
        return sign | (static_cast<uint32_t>(output_exp + 127) << 23) | (significand & kFracMask);
    }
    return sign | static_cast<uint32_t>(product >> -(power + 149));
}

inline uint32_t f32(Path op, uint32_t bits) {
    switch (op) {
    case Path::EX2:  return sfu_mufu_ex2(bits);
    case Path::LG2:  return sfu_mufu_lg2(bits);
    case Path::RCP:  return sfu_mufu_rcp(bits);
    case Path::RSQ:  return sfu_mufu_rsq(bits);
    case Path::SQRT: return sfu_mufu_sqrt(bits);
    case Path::SIN:  return sfu_mufu_sin(prescale_rz(bits));
    case Path::COS:  return sfu_mufu_cos(prescale_rz(bits));
    case Path::TANH: return sfu_mufu_tanh(bits);
    default: return 0;
    }
}

inline uint16_t f16(Path op, uint16_t bits) {
    switch (op) {
    case Path::EX2_F16:  return sfu_mufu_ex2_f16(bits);
    case Path::TANH_F16: return sfu_mufu_tanh_f16(bits);
    default: return 0;
    }
}

// Bare MUFU uses only the high word. PTX's surrounding instructions clear the output low word.
inline uint32_t f64_high(Path op, uint32_t high) {
    switch (op) {
    case Path::RCP64H: return sfu_mufu_rcp64h(high);
    case Path::RSQ64H: return sfu_mufu_rsq64h(high);
    default: return 0;
    }
}

// Unified test interface: f32 uses low 32 bits, f16 uses low 16 bits, f64 uses a full 64-bit operand.
inline uint64_t evaluate(Path op, uint64_t bits) {
    if (op == Path::RCP64H || op == Path::RSQ64H)
        return static_cast<uint64_t>(f64_high(op, static_cast<uint32_t>(bits >> 32))) << 32;
    if (op == Path::EX2_F16 || op == Path::TANH_F16)
        return f16(op, static_cast<uint16_t>(bits));
#ifdef SFU_ENABLE_BF16
    if (op == Path::EX2_BF16 || op == Path::TANH_BF16)
        return bf16_ptx::evaluate(op == Path::EX2_BF16 ? bf16_ptx::Path::EX2 : bf16_ptx::Path::TANH,
                                  static_cast<uint16_t>(bits));
#endif
    return f32(op, static_cast<uint32_t>(bits));
}

}} // namespace mufu_model::minimal
