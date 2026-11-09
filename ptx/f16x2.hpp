// Packed ex2 / tanh .f16x2: the high and low halves are processed independently.
#pragma once
#include "mufu_model.h"

namespace mufu_model { namespace f16x2_ptx {

enum class Path : int { EX2, TANH, Count };
constexpr int kPathCount = static_cast<int>(Path::Count);
constexpr const char* kPathNames[kPathCount] = {"ex2.approx.f16x2", "tanh.approx.f16x2"};

inline uint32_t evaluate(Path path, uint32_t bits) {
    const uint16_t lo = static_cast<uint16_t>(bits);
    const uint16_t hi = static_cast<uint16_t>(bits >> 16);
    const uint32_t low_result = path == Path::EX2 ? sfu_mufu_ex2_f16(lo) : sfu_mufu_tanh_f16(lo);
    const uint32_t high_result = path == Path::EX2 ? sfu_mufu_ex2_f16(hi) : sfu_mufu_tanh_f16(hi);
    return (high_result << 16) | low_result;
}

}} // namespace mufu_model::f16x2_ptx
