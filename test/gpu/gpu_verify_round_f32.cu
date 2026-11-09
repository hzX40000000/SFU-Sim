// Architecture-selected verifier for the correctly rounded rcp / sqrt (.rn .rz .rm .rp) f32 PTX paths.
#include "gpu_common.hpp"
#include "ptx/round_f32.hpp"

using namespace mufu_model;

constexpr uint32_t kAnchors[] = {
    0x00000000u, 0x00000001u, 0x00200000u, 0x003fffffu, 0x007fffffu, 0x00800000u,
    0x80200000u, 0x3f000000u, 0x3f800000u, 0x40000000u, 0x7e800000u, 0x7f000000u,
    0x7f7fffffu, 0x7f800000u, 0x7f800001u, 0x7fa00001u, 0x7fc00000u,
    0x80000000u, 0x80000001u, 0x807fffffu, 0x80800000u,
    0xbf800000u, 0xff7fffffu, 0xff800000u, 0xff800001u, 0xffc00000u
};
constexpr uint64_t kPrefix = 1ull << 16;
constexpr uint64_t kFull = 1ull << 32;

template <int Op>
__device__ __forceinline__ uint32_t hardware(uint32_t bits) {
    const float x = __uint_as_float(bits);
    float y;
    if constexpr (Op == 0) asm volatile("rcp.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 1) asm volatile("rcp.rz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 2) asm volatile("rcp.rm.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 3) asm volatile("rcp.rp.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 4) asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 5) asm volatile("sqrt.rz.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 6) asm volatile("sqrt.rm.f32 %0, %1;" : "=f"(y) : "f"(x));
    if constexpr (Op == 7) asm volatile("sqrt.rp.f32 %0, %1;" : "=f"(y) : "f"(x));
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

template <int Op>
static bool run_one(sfu_test::Context& ctx, bool smoke, const std::vector<uint32_t>& samples) {
    const uint64_t total = smoke ? kPrefix + samples.size() : kFull;
    auto input_of = [smoke, &samples](uint64_t i) -> uint64_t {
        return (!smoke || i < kPrefix) ? i : samples[i - kPrefix];
    };
    auto model_of = [](uint64_t in) -> uint64_t {
        return round_f32::evaluate(static_cast<round_f32::Path>(Op), static_cast<uint32_t>(in));
    };
    const char* name = round_f32::kPathNames[Op];
    const sfu_test::RunResult r = ctx.run_path(name, total, input_of, model_of, &launch<Op>);
    return sfu_test::report(name, r, total, 8);
}

int main(int argc, char** argv) {
    if ((argc != 2 && argc != 3) || (std::strcmp(argv[1], "smoke") && std::strcmp(argv[1], "full"))) {
        std::fprintf(stderr, "usage: %s smoke|full [ptx-path]\n", argv[0]);
        return 2;
    }
    int selected = -1;
    if (argc == 3) {
        for (int i = 0; i < round_f32::kPathCount; ++i)
            if (std::strcmp(argv[2], round_f32::kPathNames[i]) == 0) selected = i;
        if (selected < 0) {
            std::fprintf(stderr, "unknown path: %s\n", argv[2]);
            return 2;
        }
    }
    const bool smoke = std::strcmp(argv[1], "smoke") == 0;

    sfu_test::Context ctx;
    if (!ctx.arch_ok()) return 2;
    ctx.print_banner(smoke ? "ROUND F32 SMOKE" : "ROUND F32 FULL", selected < 0 ? round_f32::kPathCount : 1);
    ctx.allocate();

    std::vector<uint32_t> samples;
    if (smoke) {
        for (uint32_t anchor : kAnchors) {
            samples.push_back(anchor - 1u);
            samples.push_back(anchor);
            samples.push_back(anchor + 1u);
        }
    }
    bool ok = true;
#define RUN(OP) if (selected < 0 || selected == OP) ok = run_one<OP>(ctx, smoke, samples) && ok;
    RUN(0) RUN(1) RUN(2) RUN(3) RUN(4) RUN(5) RUN(6) RUN(7)
#undef RUN
    std::puts(selected >= 0 ? (ok ? "ROUND F32 SELECTED PASS" : "ROUND F32 SELECTED FAIL") :
              smoke ? (ok ? "ROUND F32 SMOKE PASS" : "ROUND F32 SMOKE FAIL") :
                      (ok ? "ROUND F32 PASS" : "ROUND F32 FAIL"));
    return ok ? 0 : 1;
}
