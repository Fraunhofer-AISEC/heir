#pragma once

#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <functional>
#include <iostream>

namespace bench {

struct Stats {
    size_t iterations = 0;
    double mean_ms = 0.0;
    double stddev_ms = 0.0;  // sample stddev
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

inline Stats compute_stats(std::vector<double> samples_ms) {
    Stats s;
    const size_t n = samples_ms.size();
    s.iterations = n;
    if (n == 0) {
        return s;
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    s.min_ms = samples_ms.front();
    s.max_ms = samples_ms.back();

    const double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    s.mean_ms = sum / static_cast<double>(n);

    if (n > 1) {
        double accum = 0.0;
        for (double x : samples_ms) {
            const double d = x - s.mean_ms;
            accum += d * d;
        }
        s.stddev_ms = std::sqrt(accum / static_cast<double>(n - 1)); // sample stddev
    }

    if (n % 2 == 0)
        s.median_ms = 0.5 * (samples_ms[n/2 - 1] + samples_ms[n/2]);
    else
        s.median_ms = samples_ms[n/2];

    return s;
}

// Simple "blackhole" to discourage the compiler from eliding results in benchmarks.
// For non-void results, we call consume(out). No-op for void.
inline void consume_ptr(const void* p) {
    static volatile std::uintptr_t guard = 0;
    guard ^= reinterpret_cast<std::uintptr_t>(p);
}

template <typename T>
inline void consume(const T& v) {
    consume_ptr(static_cast<const void*>(&v));
}

// Print helper (optional).
inline void print(const Stats& s, const char* label = nullptr) {
    if (label) std::cout << label << " ";
    std::cout << "runtime over " << s.iterations << " iterations (ms):\n";
    std::cout << "Mean:   " << s.mean_ms   << "\n";
    std::cout << "StdDev: " << s.stddev_ms << "\n";
    std::cout << "Median: " << s.median_ms << "\n";
    std::cout << "Min:    " << s.min_ms    << "\n";
    std::cout << "Max:    " << s.max_ms    << "\n";
}

// Generic benchmark runner.
// - F: callable
// - warmup_iters: number of warm-up calls (excluded from stats)
// - bench_iters: number of timed iterations
// - args...: arguments forwarded to the callable
// Supports both void and non-void return types.
template <typename F, typename... Args>
Stats run(F&& func, size_t warmup_iters, size_t bench_iters, Args&&... args) {
    using clock = std::chrono::steady_clock;
    using Ret = std::invoke_result_t<F, Args...>;

    // Warm-up
    for (size_t i = 0; i < warmup_iters; ++i) {
        if constexpr (std::is_void_v<Ret>) {
            std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
        } else {
            auto tmp = std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
            consume(tmp);
        }
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(bench_iters);

    // Timed iterations
    Ret last{};
    for (size_t i = 0; i < bench_iters; ++i) {
        const auto t0 = clock::now();
        if constexpr (std::is_void_v<Ret>) {
            std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
        } else {
            last = std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
        }
        const auto t1 = clock::now();

        std::chrono::duration<double, std::milli> dur = t1 - t0;
        samples_ms.push_back(dur.count());
    }

    // Prevent elision of the last result for non-void functions.
    if constexpr (!std::is_void_v<Ret>) {
        consume(last);
    }

    return compute_stats(std::move(samples_ms));
}

} // namespace bench