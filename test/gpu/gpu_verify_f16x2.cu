// Architecture-selected verifier for packed EX2 / TANH f16x2 PTX.
#include "gpu_common.hpp"
#include "ptx/f16x2.hpp"

using namespace mufu_model;

template <int Op>
__device__ __forceinline__ uint32_t hardware(uint32_t bits) {
    uint32_t result;
    if constexpr (Op == 0)
        asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(result) : "r"(bits));
    else
        asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(result) : "r"(bits));
    return result;
}

template <int Op>
__global__ void __launch_bounds__(256) hw_kernel(const uint64_t* in, uint64_t* out, uint64_t n) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t k = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < n; k += stride)
        out[k] = hardware<Op>(static_cast<uint32_t>(in[k]));
}

template <int Op>
static void launch(const uint64_t* d_in, uint64_t* d_out, uint64_t n, int blocks) {
    hw_kernel<Op><<<blocks, 256>>>(d_in, d_out, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
}

constexpr uint64_t kFull = 1ull << 32;
constexpr uint64_t kSmoke = 2ull << 16;   // every low half with high half 0, then every high half with low half 0

template <int Op>
static bool run_one(sfu_test::Context& ctx, bool smoke) {
    const uint64_t total = smoke ? kSmoke : kFull;
    auto input_of = [smoke](uint64_t i) -> uint64_t {
        if (!smoke) return i;
        return i < (1ull << 16) ? i : (i - (1ull << 16)) << 16;
    };
    auto model_of = [](uint64_t in) -> uint64_t {
        return f16x2_ptx::evaluate(static_cast<f16x2_ptx::Path>(Op), static_cast<uint32_t>(in));
    };
    const char* name = f16x2_ptx::kPathNames[Op];
    const sfu_test::RunResult r = ctx.run_path(name, total, input_of, model_of, &launch<Op>);
    return sfu_test::report(name, r, total, 8);
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "smoke") != 0 && std::strcmp(argv[1], "full") != 0)) {
        std::fprintf(stderr, "usage: %s smoke|full (build with matching ARCH)\n", argv[0]);
        return 2;
    }
    const bool smoke = std::strcmp(argv[1], "smoke") == 0;

    sfu_test::Context ctx;
    if (!ctx.arch_ok()) return 2;
    ctx.print_banner(smoke ? "F16X2 SMOKE" : "F16X2 FULL", f16x2_ptx::kPathCount);
    ctx.allocate();

    bool ok = run_one<0>(ctx, smoke);
    ok = run_one<1>(ctx, smoke) && ok;
    std::puts(smoke ? (ok ? "F16X2 SMOKE PASS" : "F16X2 SMOKE FAIL") :
                      (ok ? "F16X2 PASS" : "F16X2 FAIL"));
    return ok ? 0 : 1;
}
