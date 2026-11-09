// Shared host-side harness for the GPU verifiers.
//
// For every chunk of inputs the GPU executes the PTX instruction under test and returns its
// result. The model (the precompiled library, via ptx/) runs on the CPU for the same inputs, so the
// hardware result and the model result are computed independently and compared bit for bit.
//
// Environment: SFU_THREADS sets the number of CPU threads used for the model (default: all cores).
#pragma once
#include <cuda_runtime.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifndef SFU_TARGET_SM
#error Define SFU_TARGET_SM to the requested CUDA architecture code (for example 80 or 120)
#endif

#define CK(x) do { const cudaError_t e__ = (x); if (e__ != cudaSuccess) { \
    std::fprintf(stderr, "FAIL: CUDA %s (%s:%d)\n", cudaGetErrorString(e__), __FILE__, __LINE__); std::exit(2); \
} } while (0)

namespace sfu_test {

constexpr int kTargetMajor = SFU_TARGET_SM / 10;
constexpr int kTargetMinor = SFU_TARGET_SM % 10;
constexpr uint64_t kChunk = 1ull << 23;   // inputs per GPU launch / CPU comparison round

// Persistent worker pool: run() executes job(thread_id) on every thread (the caller is thread 0).
class ThreadPool {
public:
    explicit ThreadPool(unsigned n) : n_(n ? n : 1) {
        for (unsigned i = 1; i < n_; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lock(m_); stop_ = true; }
        start_.notify_all();
        for (auto& t : workers_) t.join();
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const { return n_; }

    void run(const std::function<void(unsigned)>& job) {
        {
            std::lock_guard<std::mutex> lock(m_);
            job_ = &job;
            pending_ = n_ - 1;
            ++generation_;
        }
        start_.notify_all();
        job(0);
        std::unique_lock<std::mutex> lock(m_);
        done_.wait(lock, [this] { return pending_ == 0; });
    }

private:
    void worker_loop(unsigned id) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(unsigned)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_);
                start_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                job = job_;
            }
            (*job)(id);
            {
                std::lock_guard<std::mutex> lock(m_);
                if (--pending_ == 0) done_.notify_all();
            }
        }
    }

    unsigned n_;
    std::mutex m_;
    std::condition_variable start_, done_;
    const std::function<void(unsigned)>* job_ = nullptr;
    uint64_t generation_ = 0;
    unsigned pending_ = 0;
    bool stop_ = false;
    std::vector<std::thread> workers_;
};

inline unsigned thread_count() {
    if (const char* env = std::getenv("SFU_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0 && v <= 4096) return static_cast<unsigned>(v);
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? hc : 4;
}

inline void split(uint64_t total, unsigned parts, unsigned id, uint64_t& lo, uint64_t& hi) {
    lo = total * id / parts;
    hi = total * (id + 1) / parts;
}

struct RunResult {
    uint64_t checked = 0, bad = 0;
    bool have = false;                       // a first mismatch was recorded
    uint64_t index = 0, input = 0, model = 0, hardware = 0;
};

struct alignas(64) Tally {
    uint64_t checked = 0, bad = 0;
    bool have = false;
    uint64_t index = 0, input = 0, model = 0, hardware = 0;
};

// Signature of the per-verifier function that runs the PTX under test on the GPU for n inputs.
using LaunchFn = void (*)(const uint64_t* d_in, uint64_t* d_out, uint64_t n, int blocks);

class Context {
public:
    Context() : pool_(thread_count()) {
        int dev = 0;
        CK(cudaGetDevice(&dev));
        CK(cudaGetDeviceProperties(&prop_, dev));
        CK(cudaDriverGetVersion(&driver_));
        CK(cudaRuntimeGetVersion(&runtime_));
        blocks_ = prop_.multiProcessorCount * 16;
    }
    ~Context() {
        if (d_in_) cudaFree(d_in_);
        if (d_out_) cudaFree(d_out_);
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // The binary is built for one architecture; refuse to run anywhere else.
    bool arch_ok() const {
        if (prop_.major == kTargetMajor && prop_.minor == kTargetMinor) return true;
        std::fprintf(stderr, "FAIL: binary targets sm_%d%d, but selected GPU is sm_%d%d; rebuild with matching ARCH\n",
                     kTargetMajor, kTargetMinor, prop_.major, prop_.minor);
        return false;
    }

    void print_banner(const char* mode, int paths) const {
        std::printf("GPU: %s sm_%d%d; build=sm_%d%d; CUDA driver=%d runtime=%d; mode: %s; paths=%d\n",
                    prop_.name, prop_.major, prop_.minor, kTargetMajor, kTargetMinor, driver_, runtime_, mode, paths);
        std::printf("model evaluated on %u CPU thread(s)\n", pool_.size());
        std::fflush(stdout);
    }

    void allocate() {
        h_in_.resize(kChunk);
        h_out_.resize(kChunk);
        CK(cudaMalloc(&d_in_, kChunk * sizeof(uint64_t)));
        CK(cudaMalloc(&d_out_, kChunk * sizeof(uint64_t)));
    }

    // Compares hardware and model for inputs input_of(0 .. total-1).
    //   input_of(index) -> uint64_t   input operand of PTX instruction (as raw bits)
    //   model_of(input) -> uint64_t   model output for that operand
    //   inject_fault: flip bit 0 of the model output at index 0 (negative control)
    template <class InputFn, class ModelFn>
    RunResult run_path(const char* name, uint64_t total, InputFn input_of, ModelFn model_of,
                       LaunchFn launch, bool inject_fault = false) {
        std::vector<Tally> tally(pool_.size());
#ifdef _WIN32
        const bool tty = _isatty(_fileno(stderr)) != 0;
#else
        const bool tty = isatty(STDERR_FILENO) != 0;
#endif
        int last_pct = -1;
        for (uint64_t begin = 0; begin < total; begin += kChunk) {
            const uint64_t n = std::min<uint64_t>(kChunk, total - begin);

            pool_.run([&](unsigned id) {
                uint64_t lo, hi;
                split(n, pool_.size(), id, lo, hi);
                for (uint64_t k = lo; k < hi; ++k) h_in_[k] = input_of(begin + k);
            });
            CK(cudaMemcpy(d_in_, h_in_.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
            launch(d_in_, d_out_, n, blocks_);
            CK(cudaMemcpy(h_out_.data(), d_out_, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            pool_.run([&](unsigned id) {
                uint64_t lo, hi;
                split(n, pool_.size(), id, lo, hi);
                Tally& t = tally[id];
                for (uint64_t k = lo; k < hi; ++k) {
                    const uint64_t in = h_in_[k];
                    uint64_t model = model_of(in);
                    if (inject_fault && begin + k == 0) model ^= 1ull;
                    const uint64_t hw = h_out_[k];
                    ++t.checked;
                    if (hw != model) {
                        ++t.bad;
                        if (!t.have) {
                            t.have = true;
                            t.index = begin + k;
                            t.input = in;
                            t.model = model;
                            t.hardware = hw;
                        }
                    }
                }
            });

            if (tty) {
                const int pct = static_cast<int>((begin + n) * 100 / total);
                if (pct != last_pct) {
                    last_pct = pct;
                    std::fprintf(stderr, "\r  %-28s %3d%%", name, pct);
                    std::fflush(stderr);
                }
            }
        }
        if (tty) std::fprintf(stderr, "\r%*s\r", 40, "");

        RunResult r;
        for (const Tally& t : tally) {
            r.checked += t.checked;
            r.bad += t.bad;
            if (t.have && (!r.have || t.index < r.index)) {
                r.have = true;
                r.index = t.index;
                r.input = t.input;
                r.model = t.model;
                r.hardware = t.hardware;
            }
        }
        return r;
    }

private:
    ThreadPool pool_;
    cudaDeviceProp prop_{};
    int driver_ = 0, runtime_ = 0, blocks_ = 1;
    std::vector<uint64_t> h_in_, h_out_;
    uint64_t* d_in_ = nullptr;
    uint64_t* d_out_ = nullptr;
};

// Prints "OK   <path> checked=N mismatch=M" (or FAIL plus the first mismatch).
inline bool report(const char* name, const RunResult& r, uint64_t total, int hex_digits) {
    const bool ok = r.checked == total && r.bad == 0;
    std::printf("%s %s checked=%llu mismatch=%llu\n", ok ? "OK  " : "FAIL", name,
                static_cast<unsigned long long>(r.checked), static_cast<unsigned long long>(r.bad));
    if (!ok && r.have)
        std::printf("  input=0x%0*llx model=0x%0*llx hardware=0x%0*llx\n",
                    hex_digits, static_cast<unsigned long long>(r.input),
                    hex_digits, static_cast<unsigned long long>(r.model),
                    hex_digits, static_cast<unsigned long long>(r.hardware));
    std::fflush(stdout);
    return ok;
}

} // namespace sfu_test
