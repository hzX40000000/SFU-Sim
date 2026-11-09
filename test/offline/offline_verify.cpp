// Compare measured PTX input/output bit patterns against the CPU model. No CUDA required.
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptx/minimal.hpp"

using namespace mufu_model;

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [hardware-reference.tsv]\n", argv[0]);
        return 2;
    }
    const char* filename = argc == 2 ? argv[1] : "test/offline/a100_sm80.tsv";
    std::ifstream data(filename);
    if (!data) {
        std::fprintf(stderr, "FAIL: cannot open hardware reference: %s\n", filename);
        return 2;
    }

    std::array<uint64_t, minimal::kPathCount> per_path{};
    uint64_t expected_rows = 0, checked = 0, mismatches = 0;
    bool header = false, device = false;
    std::string line;
    size_t line_number = 0;
    while (std::getline(data, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line.rfind("# device ", 0) == 0) { device = true; continue; }
        if (line.rfind("# rows ", 0) == 0) {
            try {
                size_t used = 0;
                const std::string count = line.substr(7);
                if (expected_rows) throw std::invalid_argument("duplicate rows");
                expected_rows = std::stoull(count, &used, 10);
                if (used != count.size() || !expected_rows) throw std::invalid_argument("invalid rows");
            } catch (...) {
                std::fprintf(stderr, "FAIL: invalid row count at line %zu\n", line_number);
                return 1;
            }
            continue;
        }
        if (line[0] == '#') continue;
        if (!header) {
            header = line == "ptx_path\tinput_bits\thardware_output_bits";
            if (!header || !device || !expected_rows) {
                std::fprintf(stderr, "FAIL: missing hardware-reference header at line %zu\n", line_number);
                return 1;
            }
            continue;
        }

        std::istringstream columns(line);
        std::string path_name, input_hex, output_hex, extra;
        if (!(columns >> path_name >> input_hex >> output_hex) || (columns >> extra)) {
            std::fprintf(stderr, "FAIL: invalid record at line %zu\n", line_number);
            return 1;
        }
        int path = 0;
        while (path < minimal::kPathCount && path_name != minimal::kPathNames[path]) ++path;
        if (path == minimal::kPathCount) {
            std::fprintf(stderr, "FAIL: unknown PTX path at line %zu: %s\n", line_number, path_name.c_str());
            return 1;
        }
        uint64_t input = 0, hardware = 0;
        try {
            size_t used_input = 0, used_output = 0;
            if (input_hex.rfind("0x", 0) != 0 || output_hex.rfind("0x", 0) != 0) throw std::invalid_argument("hex");
            input = std::stoull(input_hex, &used_input, 16);
            hardware = std::stoull(output_hex, &used_output, 16);
            if (used_input != input_hex.size() || used_output != output_hex.size()) throw std::invalid_argument("hex");
        } catch (...) {
            std::fprintf(stderr, "FAIL: invalid bit pattern at line %zu\n", line_number);
            return 1;
        }
        if ((path < 8 && ((input | hardware) >> 32)) ||
            (path >= 10 && ((input | hardware) >> 16))) {
            std::fprintf(stderr, "FAIL: bit pattern exceeds path width at line %zu\n", line_number);
            return 1;
        }
        const uint64_t model = minimal::evaluate(static_cast<minimal::Path>(path), input);
        if (model != hardware) {
            if (mismatches < 8)
                std::fprintf(stderr, "FAIL: %s input=0x%016llx model=0x%016llx hardware=0x%016llx\n",
                             path_name.c_str(), static_cast<unsigned long long>(input),
                             static_cast<unsigned long long>(model), static_cast<unsigned long long>(hardware));
            ++mismatches;
        }
        ++per_path[path];
        if (++checked > expected_rows) {
            std::fprintf(stderr, "FAIL: more records than declared in hardware reference\n");
            return 1;
        }
    }
    if (!data.eof() && data.fail()) {
        std::fprintf(stderr, "FAIL: hardware reference read error\n");
        return 1;
    }
    bool ok = header && checked == expected_rows && mismatches == 0;
    for (int path = 0; path < minimal::kPathCount; ++path) ok &= per_path[path] > 0;
    if (!ok) {
        std::fprintf(stderr, "FAIL: checked=%llu expected=%llu mismatch=%llu (all 12 paths must be present)\n",
                     static_cast<unsigned long long>(checked), static_cast<unsigned long long>(expected_rows),
                     static_cast<unsigned long long>(mismatches));
    }
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
