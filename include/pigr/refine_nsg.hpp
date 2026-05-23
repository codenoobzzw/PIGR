#pragma once
// PIGR final refinement runner for NSG (header-only).
// This header only exposes Config + run(). Implementation is in impl/refine_nsg_impl.hpp.
//
// Usage (in your minimal .cpp):
//   #define PIGR_ENABLE_REFINER
//   #include <graph/nsg.hpp>
//   int main() { pigr::refiner::nsg::Config cfg; ...; return pigr::refiner::nsg::run(cfg); }

#include <string>
#include <cstdint>

namespace pigr { namespace refiner { namespace nsg {

struct Config {
    // Required IO paths
    std::string base_path;
    std::string query_path;
    std::string gt_path;
    std::string index_path;

    // Output / logging
    std::string log_csv = "out/nsg_final.csv";

    // Threads
    int threads = 24;

    // Original set_parameters(...) args / knobs (kept as-is to minimize behavioral change)
    int param1 = 16;
    int param2 = 96;
    int param3 = 14;

    // efq knobs
    int efq_prune_init = 500;
    int efq_add_init   = 500;
    float dynamic_target_recall = 0.99f;
    int dynamic_max_efq = 2000;

    // QPS test ks
    int test_k1 = 1;
    int test_k2 = 10;
    int test_k3 = 50;
    int test_k4 = 100;

    // Batch
    int total_batches = 1;
    int batch_threads = 24;

    // Core refinement knobs
    int core_k = 20;
    float prune_ratio = 0.02f;
    int jump_max = -1;     // <=0 means auto: 2*sqrt(core_k)
    int repeat_max = 1;
    double prune_sample_ratio = 0.7;
};

inline int run(const Config& cfg);

} } } // namespace pigr::refiner::nsg

#include "impl/refine_nsg_impl.hpp"
