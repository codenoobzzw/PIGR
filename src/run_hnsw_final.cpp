#include <graph/hnsw.hpp>
#include <pigr/refine_hnsw.hpp>
#include <filesystem>
#include <string>

namespace {
std::string resolve_input_path(const std::string& path)
{
    namespace fs = std::filesystem;
    if (fs::exists(path)) {
        return path;
    }
    const std::string parent_path = std::string("../") + path;
    if (fs::exists(parent_path)) {
        return parent_path;
    }
    return path;
}

std::string resolve_output_path(const std::string& path)
{
    namespace fs = std::filesystem;
    fs::path output_path(path);
    if (!output_path.is_absolute() && fs::exists("../CMakeLists.txt")) {
        output_path = fs::path("..") / output_path;
    }
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }
    return output_path.string();
}
}

// Minimal entry point: adjust paths + knobs here (keep heavy logic inside the .hpp).
int main(int /*argc*/, char** /*argv*/)
{
    pigr::refiner::hnsw::Config cfg;

    // === Paths (edit to your local layout) ===
    cfg.base_path  = resolve_input_path("data/sift-128-euclidean.train.fvecs");
    cfg.query_path = resolve_input_path("data/sift-128-euclidean.test.fvecs");
    cfg.gt_path    = resolve_input_path("data/sift-128-euclidean.gt.ivecs");
    cfg.index_path = resolve_input_path("indices/sift-128-euclidean/hnsw/M_16_efc_96.idx");

    // === Outputs ===
    cfg.log_csv = resolve_output_path("out/hnsw_final.csv");

    // === Threads ===
    cfg.threads = 24;
    cfg.batch_threads = cfg.threads;

    // === Algorithm knobs (same meaning as your original main) ===
    cfg.param1 = 16;
    cfg.param2 = 96;
    cfg.param3 = 14;

    cfg.efq_prune_init = 500;
    cfg.efq_add_init   = 500;
    cfg.dynamic_target_recall = 0.99f;
    cfg.dynamic_max_efq = 2000;

    cfg.test_k1 = 1;
    cfg.test_k2 = 10;
    cfg.test_k3 = 50;
    cfg.test_k4 = 100;

    cfg.total_batches = 1;

    cfg.core_k = 20;
    cfg.prune_ratio = 0.02f;
    cfg.jump_max = -1;      // auto: 2*sqrt(core_k)
    cfg.repeat_max = 1;
    cfg.prune_sample_ratio = 0.7;

    cfg.operation_level_scope = 2;

    return pigr::refiner::hnsw::run(cfg);
}
