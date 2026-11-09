#include <cstdint>
#include <cstdio>
#include <vector>
#include "ptx/f16x2.hpp"
#include "ptx/round_f32.hpp"

using namespace mufu_model;

static int failures = 0;
static void expect(const char* label, uint32_t actual, uint32_t wanted) {
    if (actual != wanted) {
        std::fprintf(stderr, "FAIL %s got=0x%08x expected=0x%08x\n", label, actual, wanted);
        ++failures;
    }
}

int main() {
    using round_f32::Path;
    expect("rcp 1/3 RN", round_f32::evaluate(Path::RCP_RN, 0x40400000u), 0x3eaaaaabu);
    expect("rcp 1/3 RZ", round_f32::evaluate(Path::RCP_RZ, 0x40400000u), 0x3eaaaaaau);
    expect("rcp 1/3 RM", round_f32::evaluate(Path::RCP_RM, 0x40400000u), 0x3eaaaaaau);
    expect("rcp 1/3 RP", round_f32::evaluate(Path::RCP_RP, 0x40400000u), 0x3eaaaaabu);
    expect("rcp -1/3 RM", round_f32::evaluate(Path::RCP_RM, 0xc0400000u), 0xbeaaaaabu);
    expect("rcp -1/3 RP", round_f32::evaluate(Path::RCP_RP, 0xc0400000u), 0xbeaaaaaau);
    expect("rcp tiny RN", round_f32::evaluate(Path::RCP_RN, 0x00000001u), 0x7f800000u);
    expect("rcp tiny RZ", round_f32::evaluate(Path::RCP_RZ, 0x00000001u), 0x7f7fffffu);
    expect("rcp -tiny RM", round_f32::evaluate(Path::RCP_RM, 0x80000001u), 0xff800000u);
    expect("rcp -tiny RP", round_f32::evaluate(Path::RCP_RP, 0x80000001u), 0xff7fffffu);
    expect("rcp +infinity", round_f32::evaluate(Path::RCP_RN, 0x7f800000u), 0x00000000u);
    expect("rcp -zero", round_f32::evaluate(Path::RCP_RN, 0x80000000u), 0xff800000u);
    expect("rcp subnormal", round_f32::evaluate(Path::RCP_RN, 0x7f000000u), 0x00400000u);
    expect("sqrt 2 RN", round_f32::evaluate(Path::SQRT_RN, 0x40000000u), 0x3fb504f3u);
    expect("sqrt 2 RZ", round_f32::evaluate(Path::SQRT_RZ, 0x40000000u), 0x3fb504f3u);
    expect("sqrt 2 RM", round_f32::evaluate(Path::SQRT_RM, 0x40000000u), 0x3fb504f3u);
    expect("sqrt 2 RP", round_f32::evaluate(Path::SQRT_RP, 0x40000000u), 0x3fb504f4u);
    expect("sqrt 4", round_f32::evaluate(Path::SQRT_RN, 0x40800000u), 0x40000000u);
    expect("sqrt -zero", round_f32::evaluate(Path::SQRT_RP, 0x80000000u), 0x80000000u);
    expect("sqrt +infinity", round_f32::evaluate(Path::SQRT_RN, 0x7f800000u), 0x7f800000u);
    expect("ex2 f16x2 zero", f16x2_ptx::evaluate(f16x2_ptx::Path::EX2, 0), 0x3c003c00u);
    expect("ex2 f16x2 high", f16x2_ptx::evaluate(f16x2_ptx::Path::EX2, 0x3c000000u), 0x40003c00u);
    expect("ex2 f16x2 low", f16x2_ptx::evaluate(f16x2_ptx::Path::EX2, 0x00003c00u), 0x3c004000u);
    expect("tanh f16x2 zero", f16x2_ptx::evaluate(f16x2_ptx::Path::TANH, 0), 0u);
    // Direct calls into the precompiled model library (values from test/offline/a100_sm80.tsv).
    expect("ex2 +0", sfu_mufu_ex2(0x00000000u), 0x3f800000u);
    expect("ex2 -0", sfu_mufu_ex2(0x80000000u), 0x3f800000u);
    if (!sfu_model_version() || !sfu_model_version()[0]) {
        std::fprintf(stderr, "FAIL model library version string is empty\n");
        ++failures;
    }
    std::puts(failures ? "MODEL SELF TEST FAIL" : "MODEL SELF TEST PASS");
    return failures ? 1 : 0;
}
