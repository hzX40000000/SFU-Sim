// Generic IEEE-754 single-precision bit helpers used by the PTX-level wrappers in ptx/.
// Nothing here depends on the MUFU model data.
#pragma once
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace mufu_model {

constexpr uint32_t kSignMask = 0x80000000u;
constexpr uint32_t kExpMask  = 0x7f800000u;
constexpr uint32_t kFracMask = 0x007fffffu;
constexpr uint32_t kInfBits  = 0x7f800000u;
constexpr uint32_t kNanBits  = 0x7fffffffu;

// Index of the highest set bit; -1 for zero.
#if defined(_MSC_VER) && !defined(__clang__)
inline int highest32(uint32_t value) {
    unsigned long index;
    return _BitScanReverse(&index, value) ? static_cast<int>(index) : -1;
}
inline int highest64(uint64_t value) {
    unsigned long index;
    return _BitScanReverse64(&index, value) ? static_cast<int>(index) : -1;
}
#else
inline int highest32(uint32_t value) { return value ? 31 - __builtin_clz(value) : -1; }
inline int highest64(uint64_t value) { return value ? 63 - __builtin_clzll(value) : -1; }
#endif

inline uint32_t fp32_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float fp32_value(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// value / 2^shift rounded to nearest even.
inline uint64_t round_even(uint64_t value, int shift) {
    if (shift <= 0) return value << -shift;
    if (shift > 64) return 0;
    if (shift == 64) return value > (uint64_t{1} << 63) ? 1 : 0;  // An exact halfway case rounds to even zero.
    const uint64_t quotient = value >> shift;
    const uint64_t remainder = value & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    return quotient + (remainder > half || (remainder == half && (quotient & 1)));
}

// Convert positive integer value * 2^power to FP32, including subnormals and RNE.
inline uint32_t pack_exact_rn(uint64_t value, int power) {
    if (!value) return 0;
    const int top = highest64(value);
    int exponent = top + power;
    if (exponent > 127) return kInfBits;
    if (exponent >= -126) {
        uint64_t significand = round_even(value, top - 23);
        if (significand == (uint64_t{1} << 24)) {
            significand >>= 1;
            if (++exponent > 127) return kInfBits;
        }
        return (static_cast<uint32_t>(exponent + 127) << 23) | (static_cast<uint32_t>(significand) & kFracMask);
    }
    const int shift = -(power + 149);
    const uint64_t fraction = round_even(value, shift);
    return fraction >= (1u << 23) ? 0x00800000u : static_cast<uint32_t>(fraction);
}

// FP32 multiply (RNE) on bit patterns, preserving signed zero and infinity.
inline uint32_t mul_rn(uint32_t a, uint32_t b) {
    const uint32_t sign = (a ^ b) & kSignMask;
    const uint32_t ea = (a & kExpMask) >> 23, eb = (b & kExpMask) >> 23;
    if ((ea == 255 && (a & kFracMask)) || (eb == 255 && (b & kFracMask))) return kNanBits;
    if (ea == 255 || eb == 255) {
        if ((a & ~kSignMask) == 0 || (b & ~kSignMask) == 0) return kNanBits;
        return sign | kInfBits;
    }
    const uint32_t sa = ea ? ((1u << 23) | (a & kFracMask)) : (a & kFracMask);
    const uint32_t sb = eb ? ((1u << 23) | (b & kFracMask)) : (b & kFracMask);
    if (!sa || !sb) return sign;
    const int pa = ea ? static_cast<int>(ea) - 150 : -149;
    const int pb = eb ? static_cast<int>(eb) - 150 : -149;
    return sign | pack_exact_rn(static_cast<uint64_t>(sa) * sb, pa + pb);
}

// FP32 add (RNE) on bit patterns. The sum of two floats whose exponents differ by at most 29
// is exact in binary64, so rounding once to FP32 gives the correctly rounded result.
inline uint32_t add_rn(uint32_t a, uint32_t b) {
    if (((a & kExpMask) == kExpMask && (a & kFracMask)) || ((b & kExpMask) == kExpMask && (b & kFracMask))) return kNanBits;
    if ((a & kExpMask) == kExpMask) {
        if ((b & kExpMask) == kExpMask && ((a ^ b) & kSignMask)) return kNanBits;
        return a;
    }
    if ((b & kExpMask) == kExpMask) return b;
    return fp32_bits(static_cast<float>(static_cast<double>(fp32_value(a)) + static_cast<double>(fp32_value(b))));
}

} // namespace mufu_model
