#include <graph/nsg.hpp>
#include <pgb/refine_nsg.hpp>
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

int main(int /*argc*/, char** /*argv*/)
{
    pgb::refiner::nsg::Config cfg;

    cfg.base_path  = resolve_input_path("data/sift-128-euclidean.train.fvecs");
    cfg.query_path = resolve_input_path("data/sift-128-euclidean.test.fvecs");
    cfg.gt_path    = resolve_input_path("data/sift-128-euclidean.gt.ivecs");
    cfg.index_path = resolve_input_path("indices/sift-128-euclidean/nsg/R_64_efc_256.idx");

    cfg.log_csv = resolve_output_path("out/nsg_final.csv");

    cfg.threads = 24;
    cfg.batch_threads = cfg.threads;

    cfg.param1 = 16;
    cfg.param2 = 96;
    cfg.param3 = 20;

    cfg.efq_prune_init = 500;
    cfg.efq_add_init   = 500;
    cfg.dynamic_target_recall = 0.90f;
    cfg.dynamic_max_efq = 2000;

    cfg.test_k1 = 1;
    cfg.test_k2 = 10;
    cfg.test_k3 = 50;
    cfg.test_k4 = 100;

    cfg.total_batches = 1;

    cfg.core_k = 20;
    cfg.prune_ratio = 0.02f;
    cfg.jump_max = -1;
    cfg.repeat_max = 1;
    cfg.prune_sample_ratio = 0.2;

    return pgb::refiner::nsg::run(cfg);
}
