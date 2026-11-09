// Architecture-selected verifier for seven non-.ftz approximate f32 PTX paths.
#include "gpu_common.hpp"
#include "ptx/extended_f32.hpp"

using namespace mufu_model;

template <int Op>
__device__ __forceinline__ uint32_t hardware(uint32_t bits) {
    const float x = __uint_as_float(bits);
    float y;
    if constexpr (Op == 0) asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 1) asm volatile("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 2) asm volatile("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 3) asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 4) asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 5) asm volatile("sin.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 6) asm volatile("cos.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return __float_as_uint(y);
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

constexpr uint64_t kFullInputs = 1ull << 32;
constexpr uint64_t kSmokePrefix = 1ull << 16;

// Besides a short contiguous prefix, check each side of important boundaries and specials.
static std::vector<uint32_t> smoke_inputs() {
    constexpr uint32_t anchors[] = {
        0x00000000u, 0x00000001u, 0x00400000u, 0x007fffffu, 0x00800000u,
        0x3e22f983u, 0x3f000000u, 0x3f800000u, 0x42fc0000u,
        0x7e800000u, 0x7f000000u, 0x7f7fffffu, 0x7f800000u,
        0x7f800001u, 0x7fa00001u, 0x7fc00000u, 0x7fffffffu,
        0x80000000u, 0x80000001u, 0x80400000u, 0x807fffffu, 0x80800000u,
        0xbf800000u, 0xc2fc0000u, 0xfe800000u, 0xff7fffffu,
        0xff800000u, 0xff800001u, 0xffa00001u, 0xffc00000u, 0xffffffffu
    };
    std::vector<uint32_t> inputs;
    inputs.reserve(3 * (sizeof(anchors) / sizeof(anchors[0])));
    for (uint32_t anchor : anchors) {
        inputs.push_back(anchor - 1u);
        inputs.push_back(anchor);
        inputs.push_back(anchor + 1u);
    }
    return inputs;
}

template <int Op>
static bool run_one(sfu_test::Context& ctx, bool smoke, const std::vector<uint32_t>& samples) {
    const uint64_t total = smoke ? kSmokePrefix + samples.size() : kFullInputs;
    auto input_of = [smoke, &samples](uint64_t i) -> uint64_t {
        return (!smoke || i < kSmokePrefix) ? i : samples[i - kSmokePrefix];
    };
    auto model_of = [](uint64_t in) -> uint64_t {
        return extended_f32::evaluate(static_cast<extended_f32::Path>(Op), static_cast<uint32_t>(in));
    };
    const char* name = extended_f32::kPathNames[Op];
    const sfu_test::RunResult r = ctx.run_path(name, total, input_of, model_of, &launch<Op>);
    return sfu_test::report(name, r, total, 8);
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "smoke") != 0 && std::strcmp(argv[1], "full") != 0)) {
        std::fprintf(stderr, "usage: %s smoke|full\n", argv[0]);
        return 2;
    }
    const bool smoke = std::strcmp(argv[1], "smoke") == 0;

    sfu_test::Context ctx;
    if (!ctx.arch_ok()) return 2;
    ctx.print_banner(smoke ? "EXTENDED F32 SMOKE" : "EXTENDED F32 FULL", extended_f32::kPathCount);
    ctx.allocate();

    const std::vector<uint32_t> samples = smoke ? smoke_inputs() : std::vector<uint32_t>{};
    bool ok = true;
#define RUN(OP) ok = run_one<OP>(ctx, smoke, samples) && ok;
    RUN(0) RUN(1) RUN(2) RUN(3) RUN(4) RUN(5) RUN(6)
#undef RUN
    std::puts(smoke ? (ok ? "EXTENDED F32 SMOKE PASS" : "EXTENDED F32 SMOKE FAIL") :
                      (ok ? "EXTENDED F32 PASS" : "EXTENDED F32 FAIL"));
    return ok ? 0 : 1;
}
