/*
 * Snapdragon Profiler — matmul / cpu / eigen_baseline (Phase 3.5).
 *
 * Square fp32 matmul C = A * B using Eigen 3.4.0. Inputs A and B
 * are deterministic: generated via splitmix64 from a user-supplied
 * seed, bit-identical to `kernels/matmul/cpu/reference.py`'s
 * NumPy reference.
 *
 * Argv contract (per `kernels/matmul/cpu/DESIGN.md` M3):
 *   --size N              square matrix size (default 512)
 *   --seed N              splitmix64 seed for inputs (default 42)
 *   --validate-only       compute once, print fingerprint, exit
 *   --iterations N        compute N times before printing (default 1)
 *
 * Stdout (per DESIGN.md M3):
 *   matmul-eigen_baseline-start size=512 seed=42
 *   matmul-eigen_baseline-end fingerprint=<float> wall_us=<int>
 *
 * Fingerprint = sum(C) + frobenius_norm(C). Validated against the
 * NumPy reference within 1e-3 relative tolerance per M4/Q1.
 *
 * Determinism (per M4/Q2): same `(size, seed)` produces the same
 * fingerprint to bit-exactness across runs. The only source of
 * cross-run variance is the floating-point reduction order, which
 * Eigen's expression-template product engine fixes for a given
 * Eigen version + target arch.
 */

#define EIGEN_MPL2_ONLY  // exclude LGPL paths per third_party/eigen-3.4.0/NOTES.md

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t SPLITMIX_M1  = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t SPLITMIX_M2  = 0x94D049BB133111EBULL;

inline uint64_t splitmix64_next(uint64_t& state) {
    state += SPLITMIX_INC;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * SPLITMIX_M1;
    z = (z ^ (z >> 27)) * SPLITMIX_M2;
    z = z ^ (z >> 31);
    return z;
}

// Convert uint64 → double in [0, 1) using the top 53 bits.
// Matches reference.py::u64_to_unit_double.
inline double u64_to_unit_double(uint64_t x) {
    return (x >> 11) * (1.0 / static_cast<double>(1ULL << 53));
}

// Generate A, B as fp32 matrices, in column-major Eigen storage.
// Element i,j of A (then B) is filled in row-major *generation*
// order (i * size + j) so the splitmix sequence matches the
// reference's flat 0..2*N iteration order. The cast double→float32
// is the standard IEEE 754 round-to-nearest.
void generate_inputs(int size, uint64_t seed,
                     Eigen::MatrixXf& A, Eigen::MatrixXf& B) {
    A.resize(size, size);
    B.resize(size, size);
    uint64_t state = seed;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            uint64_t v = splitmix64_next(state);
            double d = 2.0 * u64_to_unit_double(v) - 1.0;
            A(i, j) = static_cast<float>(d);
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            uint64_t v = splitmix64_next(state);
            double d = 2.0 * u64_to_unit_double(v) - 1.0;
            B(i, j) = static_cast<float>(d);
        }
    }
}

double fingerprint(const Eigen::MatrixXf& C) {
    double s = static_cast<double>(C.sum());
    double fro = static_cast<double>(C.norm());  // Frobenius (default for matrices)
    return s + fro;
}

// argv parsing. Returns 0 on success, non-zero on parse error.
int parse_args(int argc, char** argv,
               int& size, uint64_t& seed,
               bool& validate_only, int& iterations) {
    size = 512;
    seed = 42;
    validate_only = false;
    iterations = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = std::atoi(argv[++i]);
            if (size <= 0) {
                std::fprintf(stderr, "ERROR: --size must be > 0\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--validate-only") == 0) {
            validate_only = true;
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
            if (iterations <= 0) {
                std::fprintf(stderr, "ERROR: --iterations must be > 0\n");
                return 1;
            }
        } else {
            std::fprintf(stderr, "ERROR: unknown arg: %s\n", argv[i]);
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int size, iterations;
    uint64_t seed;
    bool validate_only;
    if (parse_args(argc, argv, size, seed, validate_only, iterations) != 0) {
        return 1;
    }

    std::printf("matmul-eigen_baseline-start size=%d seed=%llu\n",
                size, static_cast<unsigned long long>(seed));
    std::fflush(stdout);

    Eigen::MatrixXf A, B;
    generate_inputs(size, seed, A, B);

    auto t0 = std::chrono::steady_clock::now();

    Eigen::MatrixXf C;
    if (validate_only) {
        // One multiply; fingerprint then exit.
        C.noalias() = A * B;
    } else {
        // N multiplies; fingerprint of the LAST result is what we
        // report. Each iteration recomputes C in place; this lets
        // PMU / variance work see N matmuls of work in one
        // dispatch.
        for (int it = 0; it < iterations; ++it) {
            C.noalias() = A * B;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    int64_t wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - t0).count();

    double fp = fingerprint(C);
    std::printf("matmul-eigen_baseline-end fingerprint=%.6e wall_us=%lld\n",
                fp, static_cast<long long>(wall_us));
    std::fflush(stdout);
    return 0;
}
