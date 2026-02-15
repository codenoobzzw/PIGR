#pragma once
// Shared utilities for PGB final refiners (header-only).
// Keep this header as "pure utility": no algorithm-specific graph includes.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

#if defined(__GLIBC__)
#include <malloc.h> // malloc_trim
#endif

namespace pgb::common {

// ======================== Pseudo Ground Truth ========================

struct PseudoGroundTruth {
    std::vector<int> data;
    size_t dim = 0;
    size_t num = 0;

    void reset(size_t n, size_t k) {
        num = n;
        dim = k;
        data.assign(num * dim, -1);
    }

    bool assign_from_flat_vector(std::vector<int>&& flat, size_t n, size_t k) {
        if (k == 0 || flat.size() != n * k) {
            return false;
        }
        data = std::move(flat);
        num = n;
        dim = k;
        return true;
    }

    bool empty() const { return dim == 0 || data.empty(); }

    const int* row(size_t idx) const {
        if (dim == 0 || idx >= num) return nullptr;
        return data.data() + idx * dim;
    }

    int* row_mut(size_t idx) {
        if (dim == 0 || idx >= num) return nullptr;
        return data.data() + idx * dim;
    }
};

// ======================== Memory trim helpers (glibc only) ========================

static inline void trim_os_memory_once() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

static inline void trim_os_memory_all_threads(int num_threads_hint = 0) {
#if defined(__GLIBC__)
    const int nt = num_threads_hint > 0 ? num_threads_hint : omp_get_max_threads();
    #pragma omp parallel num_threads(nt)
    {
        malloc_trim(0);
    }
#endif
}

// ======================== Frontier utils ========================

class PGB9Utils {
public:
    static float linear_interpolate(float x,
                                    const std::vector<float>& x_values,
                                    const std::vector<float>& y_values) {
        if (x_values.empty() || y_values.empty() || x_values.size() != y_values.size()) return 0.0f;
        if (x < x_values.front() || x > x_values.back()) return 0.0f;

        auto it = std::lower_bound(x_values.begin(), x_values.end(), x);
        size_t idx = static_cast<size_t>(std::distance(x_values.begin(), it));

        if (it == x_values.begin()) return y_values.front();

        float x0 = x_values[idx - 1];
        float x1 = x_values[idx];
        float y0 = y_values[idx - 1];
        float y1 = y_values[idx];

        if (x1 == x0) return y0;
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }

    static float calculate_std_dev(const std::vector<float>& values, float mean) {
        if (values.empty()) return 0.0f;
        float sum_squared_diff = 0.0f;
        for (float val : values) {
            float diff = val - mean;
            sum_squared_diff += diff * diff;
        }
        return std::sqrt(sum_squared_diff / static_cast<float>(values.size()));
    }

    static void sort_pairs(std::vector<float>& recalls, std::vector<float>& qps) {
        std::vector<std::pair<float, float>> paired;
        paired.reserve(recalls.size());
        for (size_t i = 0; i < recalls.size(); ++i) {
            paired.emplace_back(recalls[i], qps[i]);
        }
        std::sort(paired.begin(), paired.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < paired.size(); ++i) {
            recalls[i] = paired[i].first;
            qps[i] = paired[i].second;
        }
    }

    static void acquire_frontier(std::vector<float>& recalls, std::vector<float>& qps) {
        if (recalls.empty() || qps.empty() || recalls.size() != qps.size()) return;

        // Keep points with non-increasing QPS when recall increases (upper frontier).
        std::vector<float> f_recalls;
        std::vector<float> f_qps;
        f_recalls.reserve(recalls.size());
        f_qps.reserve(qps.size());

        float best_qps = -std::numeric_limits<float>::infinity();
        for (int i = static_cast<int>(recalls.size()) - 1; i >= 0; --i) {
            if (qps[i] > best_qps) {
                best_qps = qps[i];
                f_recalls.push_back(recalls[i]);
                f_qps.push_back(qps[i]);
            }
        }

        std::reverse(f_recalls.begin(), f_recalls.end());
        std::reverse(f_qps.begin(), f_qps.end());
        recalls.swap(f_recalls);
        qps.swap(f_qps);
    }
};

// ======================== Edge tracking (2D) ========================
// For NSG / HCNNG style graphs: edges are identified by (from, to).

class EdgeSearchTracker2D {
private:
    std::unordered_map<uint64_t, std::vector<int>> edge_to_base_points;
    std::vector<std::unordered_map<uint64_t, std::vector<int>>> thread_local_maps;

    bool filter_enabled_ = false;
    std::unordered_set<uint64_t> filtered_edges_;

    static inline uint64_t compress_edge_key(int from, int to) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(from)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(to));
    }

public:
    EdgeSearchTracker2D() = default;
    EdgeSearchTracker2D(const EdgeSearchTracker2D&) = delete;
    EdgeSearchTracker2D& operator=(const EdgeSearchTracker2D&) = delete;

    void init_thread_local_maps(int num_threads) {
        thread_local_maps.resize(num_threads);
        for (auto& local_map : thread_local_maps) {
            local_map.reserve(100000);
        }
    }

    void set_filter_edges(const std::vector<std::pair<int,int>>& edges) {
        filtered_edges_.clear();
        filtered_edges_.reserve(edges.size() * 2);
        for (const auto& e : edges) {
            filtered_edges_.insert(compress_edge_key(e.first, e.second));
        }
        filter_enabled_ = true;
        std::printf("Info: %zu\n", filtered_edges_.size());
    }

    void clear_filter() {
        filter_enabled_ = false;
        std::unordered_set<uint64_t>().swap(filtered_edges_);
    }

    void merge_thread_local_maps() {
        std::printf("Info.\n");

        size_t total_entries = 0;
        for (const auto& local_map : thread_local_maps) total_entries += local_map.size();
        edge_to_base_points.reserve(edge_to_base_points.size() + total_entries);

        for (auto& local_map : thread_local_maps) {
            for (auto& kv : local_map) {
                auto& global_vec = edge_to_base_points[kv.first];
                global_vec.insert(global_vec.end(), kv.second.begin(), kv.second.end());
            }
            std::unordered_map<uint64_t, std::vector<int>>().swap(local_map);
        }

        std::vector<std::unordered_map<uint64_t, std::vector<int>>>().swap(thread_local_maps);
        std::printf("Info: %zu\n", edge_to_base_points.size());

        size_t total_access_times = 0;
        for (const auto& entry : edge_to_base_points) total_access_times += entry.second.size();
        const double avg_access_times = edge_to_base_points.empty()
            ? 0.0
            : static_cast<double>(total_access_times) / static_cast<double>(edge_to_base_points.size());

        std::printf("Info: %zu, %.2f\n", total_access_times, avg_access_times);

        trim_os_memory_all_threads();
    }

    void record_edge_access(int from, int to, int base_point_index) {
        const uint64_t key = compress_edge_key(from, to);
        if (filter_enabled_ && filtered_edges_.find(key) == filtered_edges_.end()) return;

        const int thread_id = omp_get_thread_num();
        auto& base_points = thread_local_maps[thread_id][key];
        if (base_points.empty() || base_points.back() != base_point_index) {
            base_points.emplace_back(base_point_index);
        }
    }

    const std::vector<int>& get_base_points_for_edge(int from, int to) const {
        static std::vector<int> empty_vector;
        const uint64_t key = compress_edge_key(from, to);
        auto it = edge_to_base_points.find(key);
        return (it != edge_to_base_points.end()) ? it->second : empty_vector;
    }

    std::vector<std::pair<int, int>> get_all_accessed_edges() const {
        std::vector<std::pair<int, int>> edges;
        edges.reserve(edge_to_base_points.size());
        for (const auto& entry : edge_to_base_points) {
            int from = static_cast<int>(entry.first >> 32);
            int to   = static_cast<int>(entry.first & 0xFFFFFFFF);
            edges.emplace_back(from, to);
        }
        return edges;
    }

    void clear() {
        std::unordered_map<uint64_t, std::vector<int>>().swap(edge_to_base_points);
        trim_os_memory_all_threads();
    }

    size_t get_total_tracked_edges() const { return edge_to_base_points.size(); }

    size_t get_total_access_records() const {
        size_t total = 0;
        for (const auto& entry : edge_to_base_points) total += entry.second.size();
        return total;
    }
};

// ======================== Edge tracking (3D) ========================
// For HNSW style graphs: edges are identified by (from, to, level).

struct EdgeKey {
    int from;
    int to;
    int level;

    EdgeKey() : from(-1), to(-1), level(-1) {}
    EdgeKey(int f, int t, int l) : from(f), to(t), level(l) {}

    bool operator==(const EdgeKey& other) const {
        return from == other.from && to == other.to && level == other.level;
    }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& key) const {
        const size_t h1 = std::hash<int>()(key.from);
        const size_t h2 = std::hash<int>()(key.to);
        const size_t h3 = std::hash<int>()(key.level);
        return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
    }
};

class EdgeSearchTracker3D {
private:
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge_to_base_points;
    std::vector<std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash>> thread_local_maps;

    bool filter_enabled_ = false;
    std::unordered_set<EdgeKey, EdgeKeyHash> filtered_edges_;

public:
    EdgeSearchTracker3D() = default;
    EdgeSearchTracker3D(const EdgeSearchTracker3D&) = delete;
    EdgeSearchTracker3D& operator=(const EdgeSearchTracker3D&) = delete;

    void init_thread_local_maps(int num_threads) {
        thread_local_maps.resize(num_threads);
        for (auto& local_map : thread_local_maps) {
            local_map.reserve(100000);
        }
    }

    void set_filter_edges(const std::vector<EdgeKey>& edges) {
        filtered_edges_.clear();
        filtered_edges_.reserve(edges.size() * 2);
        for (const auto& edge : edges) filtered_edges_.insert(edge);
        filter_enabled_ = true;
        std::printf("Info: %zu\n", filtered_edges_.size());
    }

    void set_filter_edges(const std::vector<std::pair<int, int>>& edges, int level = 0) {
        std::vector<EdgeKey> expanded;
        expanded.reserve(edges.size());
        for (const auto& e : edges) expanded.emplace_back(e.first, e.second, level);
        set_filter_edges(expanded);
    }

    void clear_filter() {
        filter_enabled_ = false;
        filtered_edges_.clear();
        filtered_edges_.rehash(0);
    }

    void merge_thread_local_maps() {
        std::printf("Info.\n");

        size_t total_entries = 0;
        for (const auto& local_map : thread_local_maps) total_entries += local_map.size();
        edge_to_base_points.reserve(edge_to_base_points.size() + total_entries);

        for (auto& local_map : thread_local_maps) {
            for (auto& kv : local_map) {
                auto& global_vec = edge_to_base_points[kv.first];
                global_vec.insert(global_vec.end(), kv.second.begin(), kv.second.end());
            }
            std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash>().swap(local_map);
        }

        std::vector<std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash>>().swap(thread_local_maps);
        std::printf("Info: %zu\n", edge_to_base_points.size());

        size_t total_access_times = 0;
        for (const auto& entry : edge_to_base_points) total_access_times += entry.second.size();
        const double avg_access_times = edge_to_base_points.empty()
            ? 0.0
            : static_cast<double>(total_access_times) / static_cast<double>(edge_to_base_points.size());

        std::printf("Info: %zu, %.2f\n", total_access_times, avg_access_times);

        trim_os_memory_all_threads();
    }

    void record_edge_access(int from, int to, int level, int base_point_index) {
        EdgeKey key(from, to, level);
        if (filter_enabled_ && filtered_edges_.find(key) == filtered_edges_.end()) return;

        const int thread_id = omp_get_thread_num();
        auto& base_points = thread_local_maps[thread_id][key];

        if (base_points.empty() || base_points.back() != base_point_index) {
            base_points.emplace_back(base_point_index);
        }
    }

    const std::vector<int>& get_base_points_for_edge(int from, int to, int level = 0) const {
        static std::vector<int> empty_vector;
        EdgeKey key(from, to, level);
        auto it = edge_to_base_points.find(key);
        return (it != edge_to_base_points.end()) ? it->second : empty_vector;
    }

    std::vector<EdgeKey> get_all_accessed_edges() const {
        std::vector<EdgeKey> edges;
        edges.reserve(edge_to_base_points.size());
        for (const auto& entry : edge_to_base_points) edges.emplace_back(entry.first);
        return edges;
    }

    void clear() {
        std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash>().swap(edge_to_base_points);
        trim_os_memory_all_threads();
    }

    size_t get_total_tracked_edges() const { return edge_to_base_points.size(); }

    size_t get_total_access_records() const {
        size_t total = 0;
        for (const auto& entry : edge_to_base_points) total += entry.second.size();
        return total;
    }
};

} // namespace pgb::common
