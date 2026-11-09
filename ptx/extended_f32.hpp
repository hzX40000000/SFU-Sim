// Seven non-.ftz approximate f32 PTX paths observed on sm_80.
// These wrap the bare MUFU model with the PTX-to-SASS surrounding operations.
#pragma once
#include "minimal.hpp"

namespace mufu_model { namespace extended_f32 {

enum class Path : int { EX2, LG2, RCP, RSQ, SQRT, SIN, COS, Count };
constexpr int kPathCount = static_cast<int>(Path::Count);
constexpr const char* kPathNames[kPathCount] = {
    "ex2.approx.f32", "lg2.approx.f32", "rcp.approx.f32", "rsqrt.approx.f32",
    "sqrt.approx.f32", "sin.approx.f32", "cos.approx.f32"
};

inline uint32_t evaluate(Path op, uint32_t bits) {
    const uint32_t magnitude = bits & ~kSignMask;
    switch (op) {
    case Path::EX2: {
        // FSETP.GEU x,-126; @!P FMUL x,0.5; MUFU.EX2; @!P FMUL y,y.
        const bool scale = (bits & kSignMask) && bits > 0xc2fc0000u && magnitude <= kInfBits;
        const uint32_t argument = scale ? mul_rn(bits, 0x3f000000u) : bits;
        const uint32_t result = sfu_mufu_ex2(argument);
        return scale ? mul_rn(result, result) : result;
    }
    case Path::LG2: {
        // |x| < 2^-126: FMUL x,2^24; MUFU.LG2; FADD y,-24.
        const bool scale = magnitude < 0x00800000u;
        const uint32_t argument = scale ? mul_rn(bits, 0x4b800000u) : bits;
        const uint32_t result = sfu_mufu_lg2(argument);
        return scale ? add_rn(result, 0xc1c00000u) : result;
    }
    case Path::RCP: {
        // The two FSELs select the same factor for the input and output FMULs.
        const uint32_t factor = magnitude < 0x00800000u ? 0x4b800000u
                              : magnitude > 0x7e800000u ? 0x3e800000u : 0x3f800000u;
        return mul_rn(sfu_mufu_rcp(mul_rn(bits, factor)), factor);
    }
    case Path::RSQ: {
        const bool scale = magnitude < 0x00800000u;
        const uint32_t argument = scale ? mul_rn(bits, 0x4b800000u) : bits;
        const uint32_t result = sfu_mufu_rsq(argument);
        return scale ? mul_rn(result, 0x45800000u) : result;
    }
    case Path::SQRT: {
        const bool scale = magnitude < 0x00800000u;
        const uint32_t argument = scale ? mul_rn(bits, 0x4b800000u) : bits;
        const uint32_t result = sfu_mufu_sqrt(argument);
        return scale ? mul_rn(result, 0x39800000u) : result;
    }
    case Path::SIN: return sfu_mufu_sin(minimal::prescale_rz(bits));
    case Path::COS: return sfu_mufu_cos(minimal::prescale_rz(bits));
    default: return 0;
    }
}

}} // namespace mufu_model::extended_f32
