// PTX-accessible BF16 minimal paths on sm_120; enabled only by the sm_120 build.
#pragma once
#include "mufu_model.h"

namespace mufu_model { namespace bf16_ptx {

enum class Path : int { EX2 = 0, TANH = 1 };

inline uint16_t evaluate(Path path, uint16_t bits) {
    switch (path) {
    case Path::EX2:  return sfu_mufu_ex2_bf16(bits);
    case Path::TANH: return sfu_mufu_tanh_bf16(bits);
    }
    return 0;
}

}} // namespace mufu_model::bf16_ptx
