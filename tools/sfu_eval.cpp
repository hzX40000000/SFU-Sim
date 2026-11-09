// Offline bit-pattern evaluator; outputs a model prediction, not a hardware test.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ptx/extended_f32.hpp"
#include "ptx/f16x2.hpp"
#include "ptx/round_f32.hpp"

int main(int argc, char** argv) {
    using namespace mufu_model;
    if (argc == 2 && std::strcmp(argv[1], "--list") == 0) {
        for (int op = 0; op < minimal::kPathCount; ++op) std::puts(minimal::kPathNames[op]);
        for (int op = 0; op < extended_f32::kPathCount; ++op) std::puts(extended_f32::kPathNames[op]);
        for (int op = 0; op < f16x2_ptx::kPathCount; ++op) std::puts(f16x2_ptx::kPathNames[op]);
        for (int op = 0; op < round_f32::kPathCount; ++op) std::puts(round_f32::kPathNames[op]);
        return 0;
    }
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <ptx-path> <hex-input-bits>\n", argv[0]);
        std::fprintf(stderr, "       %s --list        print all supported path names\n", argv[0]);
        return 2;
    }
    errno = 0;
    char* end = nullptr;
    if (argv[2][0] != '0' || argv[2][1] != 'x' || !argv[2][2]) {
        std::fprintf(stderr, "input must be hexadecimal with 0x prefix\n");
        return 2;
    }
    const unsigned long long value = std::strtoull(argv[2], &end, 16);
    if (errno || !end || *end) {
        std::fprintf(stderr, "invalid bit pattern: %s\n", argv[2]);
        return 2;
    }
    const char* path = argv[1];
    for (int op = 0; op < minimal::kPathCount; ++op) {
        if (std::strcmp(path, minimal::kPathNames[op]) != 0) continue;
        const int width = op == 8 || op == 9 ? 64 : op >= 10 ? 16 : 32;
        if (width < 64 && (value >> width)) break;
        const uint64_t result = minimal::evaluate(static_cast<minimal::Path>(op), value);
        if (width == 64)
            std::printf("%s input=0x%016llx model=0x%016llx\n", path, value,
                        static_cast<unsigned long long>(result));
        else if (width == 16)
            std::printf("%s input=0x%04x model=0x%04x\n", path,
                        static_cast<unsigned>(value), static_cast<unsigned>(result));
        else
            std::printf("%s input=0x%08x model=0x%08x\n", path,
                        static_cast<unsigned>(value), static_cast<unsigned>(result));
        return 0;
    }
    if (value <= 0xffffffffull) {
        const uint32_t bits = static_cast<uint32_t>(value);
        for (int op = 0; op < extended_f32::kPathCount; ++op) {
            if (std::strcmp(path, extended_f32::kPathNames[op]) != 0) continue;
            std::printf("%s input=0x%08x model=0x%08x\n", path, bits,
                        extended_f32::evaluate(static_cast<extended_f32::Path>(op), bits));
            return 0;
        }
        for (int op = 0; op < f16x2_ptx::kPathCount; ++op) {
            if (std::strcmp(path, f16x2_ptx::kPathNames[op]) != 0) continue;
            std::printf("%s input=0x%08x model=0x%08x\n", path, bits,
                        f16x2_ptx::evaluate(static_cast<f16x2_ptx::Path>(op), bits));
            return 0;
        }
        for (int op = 0; op < round_f32::kPathCount; ++op) {
            if (std::strcmp(path, round_f32::kPathNames[op]) != 0) continue;
            std::printf("%s input=0x%08x model=0x%08x\n", path, bits,
                        round_f32::evaluate(static_cast<round_f32::Path>(op), bits));
            return 0;
        }
    }
    std::fprintf(stderr, "unsupported path or input exceeds path width: %s %s\n", path, argv[2]);
    std::fprintf(stderr, "run '%s --list' for the supported path names\n", argv[0]);
    return 2;
}
