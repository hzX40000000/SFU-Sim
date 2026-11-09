// Exhaustive GPU verifier: 12 common PTX paths, plus two BF16 paths when built for sm_120.
// Build via make ARCH=sm_80 (or sm_89, sm_120) so the path set matches the target.
//
//   full      every input encoding of every path (2^32 for f32 / f64 high word, 2^16 for f16 / bf16)
//   smoke     2^22 inputs spread over the whole 32-bit range per path (all 2^16 for f16 / bf16)
//   negative  flips one bit of one model output and checks that the comparison reports it
#include "gpu_common.hpp"
#include "ptx/minimal.hpp"

#if defined(SFU_ENABLE_BF16) && SFU_TARGET_SM != 120
#error BF16 minimal paths require ARCH=sm_120
#endif

using namespace mufu_model;

template <int Op>
__device__ __forceinline__ uint64_t hardware(uint64_t bits) {
    if constexpr (Op < 8) {
        const float x = __uint_as_float(static_cast<uint32_t>(bits));
        float y;
        if constexpr (Op == 0) asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == 1) asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == 2) asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == 3) asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == 4) asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == 5) asm volatile("sin.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == 6) asm volatile("cos.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == 7) asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
        return __float_as_uint(y);
    } else if constexpr (Op < 10) {
        const double x = __longlong_as_double(static_cast<long long>(bits));
        double y;
        if constexpr (Op == 8) asm volatile("rcp.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(x));
        else                   asm volatile("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(x));
        return static_cast<uint64_t>(__double_as_longlong(y));
    } else if constexpr (Op < 12) {
        unsigned short y;
        if constexpr (Op == 10) asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y) : "h"(static_cast<unsigned short>(bits)));
        else                    asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y) : "h"(static_cast<unsigned short>(bits)));
        return y;
#ifdef SFU_ENABLE_BF16
    } else {
        unsigned short y;
        if constexpr (Op == 12) asm volatile("ex2.approx.ftz.bf16 %0, %1;" : "=h"(y) : "h"(static_cast<unsigned short>(bits)));
        else                    asm volatile("tanh.approx.bf16 %0, %1;" : "=h"(y) : "h"(static_cast<unsigned short>(bits)));
        return y;
#endif
    }
}

template <int Op>
__global__ void __launch_bounds__(256) hw_kernel(const uint64_t* in, uint64_t* out, uint64_t n) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t k = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < n; k += stride)
        out[k] = hardware<Op>(in[k]);
}

template <int Op>
static void launch(const uint64_t* d_in, uint64_t* d_out, uint64_t n, int blocks) {
    hw_kernel<Op><<<blocks, 256>>>(d_in, d_out, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
}

enum class Mode { Full, Smoke, Negative };

template <int Op>
static bool run_one(sfu_test::Context& ctx, Mode mode) {
    const bool smoke = mode == Mode::Smoke;
    const bool negative = mode == Mode::Negative;
    // Paths 0..9 take 32-bit operands (f64: the high word); paths 10.. take 16-bit operands.
    const uint64_t total = negative ? (1ull << 16)
                         : Op >= 10 ? (1ull << 16)
                         : smoke    ? (1ull << 22)
                                    : (1ull << 32);
    auto input_of = [smoke](uint64_t i) -> uint64_t {
        const uint64_t v = (smoke && Op < 10) ? ((i * 1021ull) & 0xffffffffull) : i;
        return (Op == 8 || Op == 9) ? (v << 32) : v;     // f64: operand is {high word, 0}
    };
    auto model_of = [](uint64_t in) -> uint64_t {
        return minimal::evaluate(static_cast<minimal::Path>(Op), in);
    };
    const char* name = minimal::kPathNames[Op];
    const sfu_test::RunResult r = ctx.run_path(name, total, input_of, model_of, &launch<Op>, negative);
    if (negative) {
        const bool detected = r.checked == total && r.bad == 1 && r.have && r.input == 0 && r.model != r.hardware;
        std::printf("%s %s checked=%llu mismatch=%llu injected_input=0x%016llx\n", detected ? "DETECTED" : "FAIL", name,
                    static_cast<unsigned long long>(r.checked), static_cast<unsigned long long>(r.bad),
                    static_cast<unsigned long long>(r.input));
        return detected;
    }
    return sfu_test::report(name, r, total, 16);
}

#ifdef SFU_ENABLE_BF16
#define EACH_PATH(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13)
static_assert(minimal::kPathCount == 14, "sm_120 must verify 12 common paths and both BF16 paths");
constexpr int kNegativePathCount = 2;
#else
#define EACH_PATH(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)
static_assert(minimal::kPathCount == 12, "common targets verify the original 12 paths");
constexpr int kNegativePathCount = 1;
#endif

int main(int argc, char** argv) {
    Mode mode = Mode::Full;
    if (argc == 2 && std::strcmp(argv[1], "full") == 0) mode = Mode::Full;
    else if (argc == 2 && std::strcmp(argv[1], "smoke") == 0) mode = Mode::Smoke;
    else if (argc == 2 && std::strcmp(argv[1], "negative") == 0) mode = Mode::Negative;
    else {
        std::fprintf(stderr, "usage: %s full | smoke | negative\n", argv[0]);
        return 2;
    }

    sfu_test::Context ctx;
    if (!ctx.arch_ok()) return 2;
    ctx.print_banner(mode == Mode::Negative ? "NEGATIVE CONTROL" : mode == Mode::Smoke ? "SMOKE" : "FULL",
                     mode == Mode::Negative ? kNegativePathCount : minimal::kPathCount);
    ctx.allocate();

    bool ok = true;
    if (mode == Mode::Negative) {
        ok = run_one<0>(ctx, mode);
#ifdef SFU_ENABLE_BF16
        ok = run_one<12>(ctx, mode) && ok;
#endif
    } else {
#define RUN(OP) ok = run_one<OP>(ctx, mode) && ok;
        EACH_PATH(RUN)
#undef RUN
    }
    std::puts(mode == Mode::Negative ? (ok ? "NEGATIVE CONTROL PASS" : "NEGATIVE CONTROL FAIL") :
              mode == Mode::Smoke    ? (ok ? "SMOKE PASS" : "SMOKE FAIL") :
                                       (ok ? "PASS" : "FAIL"));
    return ok ? 0 : 1;
}
