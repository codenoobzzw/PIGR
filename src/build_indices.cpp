#include <graph/hcnng.hpp>
#include <graph/hnsw.hpp>
#include <graph/nsg.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static void ensure_parent_dir(const std::string &path)
{
    fs::path p(path);
    if (p.has_parent_path())
    {
        fs::create_directories(p.parent_path());
    }
}

int main(int argc, char **argv)
{
    std::string base_path = "data/sift-128-euclidean.train.fvecs";
    std::string out_root = "indices/sift-128-euclidean";
    int threads = 24;

    size_t hnsw_m = 16;
    size_t hnsw_efc = 96;

    size_t nsg_r = 64;
    size_t nsg_efc = 256;

    size_t hcnng_s = 7;
    size_t hcnng_t = 15;
    size_t hcnng_ls = 1250;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next_val = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--base")
            base_path = next_val(arg);
        else if (arg == "--out-root")
            out_root = next_val(arg);
        else if (arg == "--threads")
            threads = std::stoi(next_val(arg));
        else if (arg == "--hnsw-m")
            hnsw_m = static_cast<size_t>(std::stoul(next_val(arg)));
        else if (arg == "--hnsw-efc")
            hnsw_efc = static_cast<size_t>(std::stoul(next_val(arg)));
        else if (arg == "--nsg-r")
            nsg_r = static_cast<size_t>(std::stoul(next_val(arg)));
        else if (arg == "--nsg-efc")
            nsg_efc = static_cast<size_t>(std::stoul(next_val(arg)));
        else if (arg == "--hcnng-s")
            hcnng_s = static_cast<size_t>(std::stoul(next_val(arg)));
        else if (arg == "--hcnng-t")
            hcnng_t = static_cast<size_t>(std::stoul(next_val(arg)));
        else if (arg == "--hcnng-ls")
            hcnng_ls = static_cast<size_t>(std::stoul(next_val(arg)));
        else if (arg == "-h" || arg == "--help")
        {
            std::cout
                << "Usage: build_indices [options]\n"
                << "  --base <train.fvecs>        default: data/sift-128-euclidean.train.fvecs\n"
                << "  --out-root <dir>            default: indices/sift-128-euclidean\n"
                << "  --threads <int>             default: 24\n"
                << "  --hnsw-m <int>              default: 16\n"
                << "  --hnsw-efc <int>            default: 96\n"
                << "  --nsg-r <int>               default: 64\n"
                << "  --nsg-efc <int>             default: 256\n"
                << "  --hcnng-s <int>             default: 7\n"
                << "  --hcnng-t <int>             default: 15\n"
                << "  --hcnng-ls <int>            default: 1250\n";
            return 0;
        }
        else
        {
            std::cerr << "Unknown option: " << arg << "\n";
            return 2;
        }
    }

    if (!fs::exists(base_path))
    {
        std::cerr << "Base dataset not found: " << base_path << "\n";
        return 1;
    }

    std::cout << "Loading base dataset: " << base_path << "\n";
    anns::DataSetWrapper<float> base(base_path);

    const std::string hnsw_idx = out_root + "/hnsw/M_" + std::to_string(hnsw_m) + "_efc_" + std::to_string(hnsw_efc) + ".idx";
    const std::string nsg_idx = out_root + "/nsg/R_" + std::to_string(nsg_r) + "_efc_" + std::to_string(nsg_efc) + ".idx";
    const std::string hcnng_idx = out_root + "/hcnng/s_" + std::to_string(hcnng_s) + "_T_" + std::to_string(hcnng_t) + "_Ls_" + std::to_string(hcnng_ls) + ".idx";

    ensure_parent_dir(hnsw_idx);
    ensure_parent_dir(nsg_idx);
    ensure_parent_dir(hcnng_idx);

    {
        std::cout << "[HNSW] build M=" << hnsw_m << " efc=" << hnsw_efc << "\n";
        anns::graph::HNSW<float, anns::metrics::euclidean> index(hnsw_m, hnsw_efc);
        index.set_num_threads(static_cast<size_t>(threads));
        index.build(base);
        index.save(hnsw_idx);
        std::cout << "[HNSW] saved: " << hnsw_idx << "\n";
    }

    {
        std::cout << "[NSG] build R=" << nsg_r << " efc=" << nsg_efc << "\n";
        anns::graph::NSG<float, anns::metrics::euclidean> index(nsg_r, nsg_efc);
        index.set_num_threads(static_cast<size_t>(threads));
        index.build(base);
        index.save(nsg_idx);
        std::cout << "[NSG] saved: " << nsg_idx << "\n";
    }

    {
        std::cout << "[HCNNG] build s=" << hcnng_s << " T=" << hcnng_t << " Ls=" << hcnng_ls << "\n";
        anns::graph::HCNNG<float, anns::metrics::euclidean> index(hcnng_t, hcnng_ls, hcnng_s);
        index.set_num_threads(static_cast<size_t>(threads));
        index.build(base);
        index.save(hcnng_idx);
        std::cout << "[HCNNG] saved: " << hcnng_idx << "\n";
    }

    std::cout << "All indices built successfully.\n";
    return 0;
}
