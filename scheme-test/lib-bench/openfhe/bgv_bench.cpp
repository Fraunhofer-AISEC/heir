#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include "openfhe.h"  // If this fails in your workspace, use: #include "src/pke/include/openfhe.h"
#include "scheme-test/bench/bench.hpp"

using namespace lbcrypto;

// Simple CLI parsing
struct Cli {
    std::vector<int> depths{1, 2, 3}; // default depths
    size_t warmup = 5;
    size_t iters  = 50;
    uint64_t ptm  = 65537; // plaintext modulus
    int rotateStep = 1;    // slots to rotate by (positive = left, negative = right)
};

static std::vector<int> split_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty())
            out.push_back(std::stoi(item));
    }
    return out;
}

static bool starts_with(const std::string& s, const std::string& p) {
    return s.rfind(p, 0) == 0;
}

static Cli parse_cli(int argc, char** argv) {
    Cli cli;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (starts_with(a, "--depths=")) {
            cli.depths = split_ints(a.substr(std::string("--depths=").size()));
        } else if (starts_with(a, "--warmup=")) {
            cli.warmup = static_cast<size_t>(std::stoul(a.substr(std::string("--warmup=").size())));
        } else if (starts_with(a, "--iters=")) {
            cli.iters = static_cast<size_t>(std::stoul(a.substr(std::string("--iters=").size())));
        } else if (starts_with(a, "--ptm=")) {
            cli.ptm = static_cast<uint64_t>(std::stoull(a.substr(std::string("--ptm=").size())));
        } else if (starts_with(a, "--rotate=")) {
            cli.rotateStep = std::stoi(a.substr(std::string("--rotate=").size()));
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: bgv_bench [--depths=1,2,3] [--warmup=5] [--iters=50] [--ptm=65537] [--rotate=1]\n";
            std::exit(0);
        }
    }
    // Deduplicate/sort depths and remove non-positive
    std::sort(cli.depths.begin(), cli.depths.end());
    cli.depths.erase(std::remove_if(cli.depths.begin(), cli.depths.end(), [](int d){ return d <= 0; }),
                     cli.depths.end());
    if (cli.depths.empty()) cli.depths = {1, 2, 3};
    if (cli.rotateStep == 0) cli.rotateStep = 1; // avoid 0 rotation (no-op)
    return cli;
}

struct Row {
    int depth;
    bench::Stats add;
    bench::Stats sub;
    bench::Stats mul;
    bench::Stats relin;
    bench::Stats modred;
    bench::Stats rotate;
};

static void print_table(const std::string& title, const std::vector<Row>& rows, char which) {
    // which: 'a' add, 's' sub, 'm' mul, 'r' relin, 'd' modreduce, 'o' rotate
    std::cout << "\n" << title << " (ms)\n";
    std::cout << "| depth | mean | stddev | median | min | max | iters |\n";
    std::cout << "|------:|-----:|-------:|-------:|----:|----:|------:|\n";
    for (const auto& r : rows) {
        const bench::Stats* s = nullptr;
        switch (which) {
            case 'a': s = &r.add;    break;
            case 's': s = &r.sub;    break;
            case 'm': s = &r.mul;    break;
            case 'r': s = &r.relin;  break;
            case 'd': s = &r.modred; break;
            case 'o': s = &r.rotate; break;
            default:  s = &r.mul;    break;
        }
        std::cout << "| " << r.depth
                  << " | " << s->mean_ms
                  << " | " << s->stddev_ms
                  << " | " << s->median_ms
                  << " | " << s->min_ms
                  << " | " << s->max_ms
                  << " | " << s->iterations
                  << " |\n";
    }
}

int main(int argc, char** argv) {
    auto cli = parse_cli(argc, argv);

    std::vector<Row> results;
    results.reserve(cli.depths.size());

    // Fixed input data (adjust length/batch-size as needed)
    std::vector<int64_t> v1{1,2,3,4,5,6,7,8};
    std::vector<int64_t> v2{8,7,6,5,4,3,2,1};

    for (int depth : cli.depths) {
        std::cout << "Setting up BGV with multiplicative depth = " << depth
                  << ", ptm = " << cli.ptm
                  << ", rotate = " << cli.rotateStep << "\n";

        // 1) Create BGV context for this depth
        CCParams<CryptoContextBGVRNS> params;
        params.SetPlaintextModulus(cli.ptm);
        params.SetMultiplicativeDepth(depth);
        // Optional: params.SetBatchSize(1024);

        CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
        cc->Enable(PKE);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);

        // 2) Keys
        auto kp = cc->KeyGen();
        cc->EvalMultKeyGen(kp.secretKey);
        // Rotation keys for requested step (include both directions)
        std::vector<int> rots{cli.rotateStep};
        if (cli.rotateStep != 0) rots.push_back(-cli.rotateStep);
        cc->EvalRotateKeyGen(kp.secretKey, rots);

        // 3) Encrypt sample plaintexts
        auto pt1 = cc->MakePackedPlaintext(v1);
        auto pt2 = cc->MakePackedPlaintext(v2);
        auto ctA = cc->Encrypt(kp.publicKey, pt1);
        auto ctB = cc->Encrypt(kp.publicKey, pt2);

        const size_t warmup = cli.warmup;
        const size_t iters  = cli.iters;
        const size_t total  = warmup + iters;

        Row row{depth, {}, {}, {}, {}, {}, {}};

        // 4) EvalAdd timing
        row.add = bench::run(
            [&]() -> Ciphertext<DCRTPoly> {
                return cc->EvalAdd(ctA, ctB);
            },
            warmup, iters
        );

        // 5) EvalSub timing
        row.sub = bench::run(
            [&]() -> Ciphertext<DCRTPoly> {
                return cc->EvalSub(ctA, ctB);
            },
            warmup, iters
        );

        // 6) EvalMult timing
        row.mul = bench::run(
            [&]() -> Ciphertext<DCRTPoly> {
                return cc->EvalMult(ctA, ctB);
            },
            warmup, iters
        );

        // 7) Precompute mults for Relin timing (to isolate relin)
        std::vector<Ciphertext<DCRTPoly>> ctMul(total);
        for (size_t i = 0; i < total; ++i) {
            ctMul[i] = cc->EvalMult(ctA, ctB);
        }
        struct RelinFunctor {
            CryptoContext<DCRTPoly> cc;
            std::vector<Ciphertext<DCRTPoly>>* inputs;
            size_t idx = 0;
            Ciphertext<DCRTPoly> operator()() {
                auto& in = (*inputs)[idx++];
                return cc->Relinearize(in);
            }
        } relinOp{cc, &ctMul};
        row.relin = bench::run(relinOp, warmup, iters);

        // 8) Precompute relins for ModReduce timing (to isolate modreduce)
        std::vector<Ciphertext<DCRTPoly>> ctRelin(total);
        for (size_t i = 0; i < total; ++i) {
            ctRelin[i] = cc->Relinearize(ctMul[i]);
        }
        struct ModReduceFunctor {
            CryptoContext<DCRTPoly> cc;
            std::vector<Ciphertext<DCRTPoly>>* inputs;
            size_t idx = 0;
            Ciphertext<DCRTPoly> operator()() {
                auto& in = (*inputs)[idx++];
                return cc->ModReduce(in);
            }
        } modReduceOp{cc, &ctRelin};
        row.modred = bench::run(modReduceOp, warmup, iters);

        // 9) EvalRotate timing
        row.rotate = bench::run(
            [&]() -> Ciphertext<DCRTPoly> {
                return cc->EvalRotate(ctA, cli.rotateStep);
            },
            warmup, iters
        );

        results.push_back(std::move(row));
    }

    // Print summaries as tables
    print_table("BGV EvalAdd", results, 'a');
    print_table("BGV EvalSub", results, 's');
    print_table("BGV EvalMult", results, 'm');
    print_table("BGV Relinearize", results, 'r');
    print_table("BGV ModReduce", results, 'd');
    print_table("BGV EvalRotate", results, 'o');

    return 0;
}