// Candidate bit-exact semantics for rcp/sqrt.rn/rz/rm/rp.f32 without .ftz on sm_80.
// This models the final rounded PTX result, not the compiler's iterative SASS sequence.
// Uses exact integer arithmetic only; it does not depend on the MUFU model library.
#pragma once
#include "mufu_bits.hpp"

namespace mufu_model { namespace round_f32 {

enum class Path : int { RCP_RN, RCP_RZ, RCP_RM, RCP_RP,
                        SQRT_RN, SQRT_RZ, SQRT_RM, SQRT_RP, Count };
constexpr int kPathCount = static_cast<int>(Path::Count);
constexpr const char* kPathNames[kPathCount] = {
    "rcp.rn.f32", "rcp.rz.f32", "rcp.rm.f32", "rcp.rp.f32",
    "sqrt.rn.f32", "sqrt.rz.f32", "sqrt.rm.f32", "sqrt.rp.f32"
};

enum class Mode : int { RN, RZ, RM, RP };
inline bool increment(Mode mode, bool negative, uint64_t quotient,
                       uint64_t remainder, uint64_t divisor) {
    if (!remainder) return false;
    if (mode == Mode::RP) return !negative;
    if (mode == Mode::RM) return negative;
    if (mode == Mode::RZ) return false;
    return (remainder << 1) > divisor ||
           ((remainder << 1) == divisor && (quotient & 1u));
}

// Positive integer square root and exact remainder; n has at most 48 bits.
inline uint32_t isqrt48(uint64_t n, uint64_t& remainder) {
    uint64_t root = 0, rem = 0;
    for (int bit = 46; bit >= 0; bit -= 2) {
        rem = (rem << 2) | ((n >> bit) & 3u);
        const uint64_t trial = (root << 2) | 1u;
        root <<= 1;
        if (rem >= trial) { rem -= trial; ++root; }
    }
    remainder = rem;
    return static_cast<uint32_t>(root);
}

inline uint32_t reciprocal(uint32_t bits, Mode mode) {
    const uint32_t sign = bits & kSignMask;
    const uint32_t magnitude = bits & ~kSignMask;
    if (magnitude > kInfBits) return kNanBits;
    if (magnitude == kInfBits) return sign;
    if (!magnitude) return sign | kInfBits;

    const uint32_t exponent_bits = magnitude >> 23;
    const uint32_t significand = (magnitude & kFracMask) | (exponent_bits ? (1u << 23) : 0u);
    const int power = exponent_bits ? static_cast<int>(exponent_bits) - 150 : -149;
    const int top = highest32(significand);
    const bool exact_power_of_two = significand == (1u << top);
    const int exponent = -power - top - (exact_power_of_two ? 0 : 1);
    const bool negative = sign != 0;
    if (exponent > 127) {
        const bool infinity = mode == Mode::RN || (mode == Mode::RP && !negative) ||
                              (mode == Mode::RM && negative);
        return sign | (infinity ? kInfBits : 0x7f7fffffu);
    }
    const bool subnormal = exponent < -126;
    const int shift = subnormal ? 149 - power : 23 - exponent - power;
    const uint64_t numerator = uint64_t{1} << shift;
    uint64_t quotient = numerator / significand;
    const uint64_t remainder = numerator % significand;
    quotient += increment(mode, negative, quotient, remainder, significand);
    if (subnormal)
        return sign | (quotient >= (1u << 23) ? 0x00800000u : static_cast<uint32_t>(quotient));
    if (quotient >= (uint64_t{1} << 24)) {
        if (exponent == 127) return sign | kInfBits;
        return sign | (static_cast<uint32_t>(exponent + 128) << 23);
    }
    return sign | (static_cast<uint32_t>(exponent + 127) << 23) |
           (static_cast<uint32_t>(quotient) & kFracMask);
}

inline uint32_t square_root(uint32_t bits, Mode mode) {
    const uint32_t magnitude = bits & ~kSignMask;
    if (magnitude > kInfBits) return kNanBits;
    if (!magnitude) return bits;  // Preserve -0.
    if (bits & kSignMask) return kNanBits;
    if (magnitude == kInfBits) return kInfBits;

    const uint32_t exponent_bits = magnitude >> 23;
    const uint32_t significand = (magnitude & kFracMask) | (exponent_bits ? (1u << 23) : 0u);
    const int power = exponent_bits ? static_cast<int>(exponent_bits) - 150 : -149;
    const int top = highest32(significand);
    const int input_exp = top + power;
    const int exponent = input_exp >= 0 ? input_exp / 2 : (input_exp - 1) / 2;
    const int shift = power + 46 - 2 * exponent;
    const uint64_t scaled = static_cast<uint64_t>(significand) << shift;
    uint64_t remainder = 0;
    uint32_t quotient = isqrt48(scaled, remainder);
    const uint64_t twice_root_plus_one = 2ull * quotient + 1;
    const bool up = mode == Mode::RP ? remainder != 0 :
                    mode == Mode::RN && 4ull * scaled > twice_root_plus_one * twice_root_plus_one;
    quotient += up;
    if (quotient == (1u << 24)) return static_cast<uint32_t>(exponent + 128) << 23;
    return (static_cast<uint32_t>(exponent + 127) << 23) | (quotient & kFracMask);
}

inline uint32_t evaluate(Path path, uint32_t bits) {
    const int index = static_cast<int>(path);
    const Mode mode = static_cast<Mode>(index & 3);
    return index < 4 ? reciprocal(bits, mode) : square_root(bits, mode);
}

}} // namespace mufu_model::round_f32
