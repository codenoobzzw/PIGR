
#include <graph/hnsw.hpp>
#include <iostream>
#include <fstream>
#include <utility>
#include <utils/recall.hpp>
#include <utils/timer.hpp>
#include <statistic/judgeall.hpp>

#include <iomanip>
#ifndef _WIN32
#include <unistd.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h> // malloc_trim
#endif
#include <queue>
#include <functional>
#include <sstream>
#include <atomic>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <algorithm> // for std::find
#include <random>
#include <cstdlib> // for rand()
#include <ctime> // for time()
#include <numeric> // for std::iota
#include <climits> // for INT_MAX
#include <cmath>
#include <limits>
#include <mutex> // for fine-grained locks
#include <omp.h> // OpenMPsupport
#include <tuple>

#include <pigr/common.hpp>

namespace pigr { namespace refiner { namespace hnsw {


using namespace std;
using namespace anns;
using namespace anns::utils;


using pigr::common::PseudoGroundTruth;
using pigr::common::PIGR9Utils;
using pigr::common::trim_os_memory_once;
using pigr::common::trim_os_memory_all_threads;
using EdgeKey = pigr::common::EdgeKey;
using EdgeKeyHash = pigr::common::EdgeKeyHash;
using EdgeSearchTracker = pigr::common::EdgeSearchTracker3D;
using IndexType = graph::HNSW<float, metrics::euclidean>;
using MultiLevelEdgeStats = std::vector<std::vector<std::vector<std::pair<int, int>>>>;

static inline void record_tracker_access(EdgeSearchTracker *tracker, int from, int to, int level, int query_point_index);

// === HNSW helper utilities (base-layer-focused access) ===
static inline int get_max_level(const IndexType &index)
{
    return index.max_level_;
}

static inline int get_node_level(const IndexType &index, int node_id)
{
    if (node_id < 0 || node_id >= static_cast<int>(index.element_levels_.size()))
    {
        return -1;
    }
    return index.element_levels_[node_id];
}

static inline std::vector<int> *get_neighbors_mut(IndexType &index, int node_id, int level)
{
    if (node_id < 0 || node_id >= static_cast<int>(index.link_lists_.size()))
    {
        return nullptr;
    }
    auto &levels = index.link_lists_[node_id];
    if (level < 0 || level >= static_cast<int>(levels.size()))
    {
        return nullptr;
    }
    return &levels[level];
}

static inline const std::vector<int> *get_neighbors_const(const IndexType &index, int node_id, int level)
{
    if (node_id < 0 || node_id >= static_cast<int>(index.link_lists_.size()))
    {
        return nullptr;
    }
    const auto &levels = index.link_lists_[node_id];
    if (level < 0 || level >= static_cast<int>(levels.size()))
    {
        return nullptr;
    }
    return &levels[level];
}

static inline size_t get_neighbor_count(const IndexType &index, int node_id, int level)
{
    if (const auto *neighbors = get_neighbors_const(index, node_id, level))
    {
        return neighbors->size();
    }
    return 0;
}

static inline std::vector<int> *get_base_neighbors_mut(IndexType &index, int node_id)
{
    return get_neighbors_mut(index, node_id, 0);
}

static inline const std::vector<int> *get_base_neighbors_const(const IndexType &index, int node_id)
{
    return get_neighbors_const(index, node_id, 0);
}

static inline int find_start_entry(IndexType &index,
                                   const float *query_vec,
                                   float &entry_dist,
                                   size_t &comparison_counter,
                                   EdgeSearchTracker *tracker = nullptr,
                                   int query_point_index = -1,
                                   std::vector<std::tuple<int, int, int, bool>> *level_events = nullptr)
{
    const int enterpoint = index.enterpoint_node_;
    if (enterpoint < 0 || enterpoint >= index.base_.num_)
    {
        entry_dist = std::numeric_limits<float>::infinity();
        return enterpoint;
    }

    entry_dist = metrics::euclidean(query_vec, index.base_[enterpoint], index.base_.dim_);
    comparison_counter++;
    int curr = enterpoint;
    int max_level = get_node_level(index, enterpoint);

    for (int level = max_level; level > 0; --level)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            const auto *neighbors = get_neighbors_const(index, curr, level);
            if (!neighbors)
            {
                continue;
            }
            for (int neighbor_id : *neighbors)
            {
                if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                {
                    continue;
                }
                float dist = metrics::euclidean(query_vec, index.base_[neighbor_id], index.base_.dim_);
                comparison_counter++;
                bool adopted = dist < entry_dist;
                if (level_events)
                {
                    level_events->emplace_back(curr, neighbor_id, level, adopted);
                }
                if (adopted)
                {
                    record_tracker_access(tracker, curr, neighbor_id, level, query_point_index);
                    entry_dist = dist;
                    curr = neighbor_id;
                    changed = true;
                }
            }
        }
    }

    return curr;
}

static inline void record_tracker_access(EdgeSearchTracker *tracker, int from, int to, int level, int query_point_index)
{
    if (!tracker || query_point_index < 0)
    {
        return;
    }
    tracker->record_edge_access(from, to, level, query_point_index);
}

// - edge
class TrackedSearchExecutor
{
private:
    IndexType &index;
    EdgeSearchTracker &tracker;

public:
    TrackedSearchExecutor(IndexType &idx, EdgeSearchTracker &track)
        : index(idx), tracker(track) {}

    // - NSG Branch and Bound ( )
    long long execute_search_with_tracking(int query_point_index, int base_point_index,
                                           int efq, int k)
    {

        // NSG Branch and Bound
        std::priority_queue<std::pair<float, int>> top_candidates;
        std::priority_queue<std::pair<float, int>> candidate_set;
        std::vector<bool> mass_visited(index.base_.num_, false);
        size_t comparison = 0;

        const float *query_vec = index.base_[query_point_index];
        float init_dist = 0.0f;
        int entry_point = find_start_entry(index, query_vec, init_dist, comparison, &tracker, query_point_index);
        if (entry_point < 0 || entry_point >= index.base_.num_)
        {
            return comparison;
        }
        top_candidates.emplace(init_dist, entry_point); // max heap
        candidate_set.emplace(-init_dist, entry_point); // min heap
        mass_visited[entry_point] = true;

        // ( Pruning )
        long long local_adopted = 0;
        long long local_comparisons = 0;

        /// @brief Branch and Bound Algorithm
        float low_bound = init_dist;
        while (candidate_set.size())
        {
            auto curr_el_pair = candidate_set.top();
            if (-curr_el_pair.first > low_bound && top_candidates.size() == efq)
                break;
            candidate_set.pop();
            int curr_node_id = curr_el_pair.second;

            const auto *neighbors = get_base_neighbors_const(index, curr_node_id);
            if (!neighbors || neighbors->empty())
            {
                continue;
            }

            // 
            for (int neighbor_id : *neighbors)
            {
                if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                    continue;
                if (!mass_visited[neighbor_id])
                {
                    mass_visited[neighbor_id] = true;
                    float dd = metrics::euclidean(index.base_[query_point_index], index.base_[neighbor_id], index.base_.dim_);
                    comparison++;
                    local_comparisons++;
                    /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
                    if (top_candidates.top().first > dd || top_candidates.size() < efq)
                    {
                        candidate_set.emplace(-dd, neighbor_id);
                        top_candidates.emplace(dd, neighbor_id);
                        if (top_candidates.size() > efq) // give up farest result so far
                            top_candidates.pop();
                        if (top_candidates.size())
                            low_bound = top_candidates.top().first;
                        local_adopted++;
                        // adopted edge
                        tracker.record_edge_access(curr_node_id, neighbor_id, 0, query_point_index);
                    }
                    // 
                }
            }
        }

        return comparison;
    }

    // (for )- edge
    long long execute_search_without_edge(const float *query_data, int base_point_index,
                                          int efq, int k, vector<int> &results,
                                          const unordered_set<pair<int, int>, ::PairHash> &disabled_edges)
    {
        // 
        results.clear();

        // NSG Branch and Bound
        std::priority_queue<std::pair<float, int>> top_candidates;
        std::priority_queue<std::pair<float, int>> candidate_set;
        std::vector<bool> mass_visited(index.base_.num_, false);
        size_t comparison = 0;

        float init_dist = 0.0f;
        int entry_point = find_start_entry(index, query_data, init_dist, comparison);
        if (entry_point < 0 || entry_point >= index.base_.num_)
        {
            return comparison;
        }
        top_candidates.emplace(init_dist, entry_point); // max heap
        candidate_set.emplace(-init_dist, entry_point); // min heap
        mass_visited[entry_point] = true;

        /// @brief Branch and Bound Algorithm
        float low_bound = init_dist;
        while (candidate_set.size())
        {
            auto curr_el_pair = candidate_set.top();
            if (-curr_el_pair.first > low_bound && top_candidates.size() == efq)
                break;
            candidate_set.pop();
            int curr_node_id = curr_el_pair.second;

            const auto *neighbors = get_base_neighbors_const(index, curr_node_id);
            if (!neighbors || neighbors->empty())
            {
                continue;
            }

            // 
            for (int neighbor_id : *neighbors)
            {
                // edge
                if (disabled_edges.count({curr_node_id, neighbor_id}) > 0)
                {
                    continue;
                }

                if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                    continue;

                if (!mass_visited[neighbor_id])
                {
                    mass_visited[neighbor_id] = true;
                    float dd = metrics::euclidean(query_data, index.base_[neighbor_id], index.base_.dim_);
                    comparison++;

                    /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
                    if (top_candidates.top().first > dd || top_candidates.size() < efq)
                    {
                        candidate_set.emplace(-dd, neighbor_id);
                        top_candidates.emplace(dd, neighbor_id);
                        if (top_candidates.size() > efq) // give up farest result so far
                            top_candidates.pop();
                        if (top_candidates.size())
                            low_bound = top_candidates.top().first;
                    }
                }
            }
        }

        return comparison;
    }

    // : ( edge )
    long long execute_search_baseline_fast(const float *query_data, int base_point_index,
                                           int efq, int k, vector<int> &results)
    {
        // 
        results.clear();

        // NSG Branch and Bound
        std::priority_queue<std::pair<float, int>> top_candidates; // max heap
        std::priority_queue<std::pair<float, int>> candidate_set;
        // visited
        thread_local std::vector<uint8_t> visited_buffer;
        if (visited_buffer.size() != (size_t)index.base_.num_)
            visited_buffer.resize(index.base_.num_);
        std::fill(visited_buffer.begin(), visited_buffer.end(), 0);

        size_t comparison = 0;
        const int num = index.base_.num_;
        const int dim = index.base_.dim_;
        float init_dist = 0.0f;
        int enter = find_start_entry(index, query_data, init_dist, comparison);
        if (enter < 0 || enter >= num)
            return comparison;

        top_candidates.emplace(init_dist, enter);
        candidate_set.emplace(-init_dist, enter);
        visited_buffer[enter] = 1;

        float low_bound = init_dist;
        while (!candidate_set.empty())
        {
            auto curr_el_pair = candidate_set.top();
            if (-curr_el_pair.first > low_bound && (int)top_candidates.size() == efq)
                break;
            candidate_set.pop();
            int curr = curr_el_pair.second;

            const auto *neighbors = get_base_neighbors_const(index, curr);
            if (!neighbors || neighbors->empty())
                continue;

            for (int nb : *neighbors)
            {
                if (nb < 0 || nb >= num)
                    continue;
                if (!visited_buffer[nb])
                {
                    visited_buffer[nb] = 1;
                    float dd = metrics::euclidean(query_data, index.base_[nb], dim);
                    comparison++;
                    if (top_candidates.empty() || top_candidates.top().first > dd || (int)top_candidates.size() < efq)
                    {
                        candidate_set.emplace(-dd, nb);
                        top_candidates.emplace(dd, nb);
                        if ((int)top_candidates.size() > efq)
                            top_candidates.pop();
                        if (!top_candidates.empty())
                            low_bound = top_candidates.top().first;
                    }
                }
            }
        }
        // ( k )
        vector<pair<float, int>> final_candidates;
        while (!top_candidates.empty())
        {
            auto [dist, node] = top_candidates.top();
            top_candidates.pop();
            final_candidates.emplace_back(dist, node);
        }
        sort(final_candidates.begin(), final_candidates.end());
        results.reserve(std::min(k, (int)final_candidates.size()));
        for (const auto &candidate : final_candidates)
        {
            if (base_point_index >= 0 && candidate.second == base_point_index)
            {
                continue;
            }
            results.push_back(candidate.second);
            if ((int)results.size() >= k)
                break;
        }

        return comparison;
    }

    // : edge (from->to)
    long long execute_search_disable_single_edge_fast(const float *query_data, int base_point_index,
                                                      int efq, int k, vector<int> &results, int from, int to)
    {
        // 
        results.clear();

        // NSG Branch and Bound
        std::priority_queue<std::pair<float, int>> top_candidates; // max heap
        std::priority_queue<std::pair<float, int>> candidate_set;
        // visited
        thread_local std::vector<uint8_t> visited_buffer;
        if (visited_buffer.size() != (size_t)index.base_.num_)
            visited_buffer.resize(index.base_.num_);
        std::fill(visited_buffer.begin(), visited_buffer.end(), 0);

        size_t comparison = 0;
        const int num = index.base_.num_;
        const int dim = index.base_.dim_;
        float init_dist = 0.0f;
        int enter = find_start_entry(index, query_data, init_dist, comparison);
        if (enter < 0 || enter >= num)
            return comparison;

        top_candidates.emplace(init_dist, enter);
        candidate_set.emplace(-init_dist, enter);
        visited_buffer[enter] = 1;

        float low_bound = init_dist;
        while (!candidate_set.empty())
        {
            auto curr_el_pair = candidate_set.top();
            if (-curr_el_pair.first > low_bound && (int)top_candidates.size() == efq)
                break;
            candidate_set.pop();
            int curr = curr_el_pair.second;

            const auto *neighbors = get_base_neighbors_const(index, curr);
            if (!neighbors || neighbors->empty())
                continue;

            for (int nb : *neighbors)
            {
                // edge (from->to)
                if (curr == from && nb == to)
                    continue;
                if (nb < 0 || nb >= num)
                    continue;
                if (!visited_buffer[nb])
                {
                    visited_buffer[nb] = 1;
                    float dd = metrics::euclidean(query_data, index.base_[nb], dim);
                    comparison++;
                    if (top_candidates.empty() || top_candidates.top().first > dd || (int)top_candidates.size() < efq)
                    {
                        candidate_set.emplace(-dd, nb);
                        top_candidates.emplace(dd, nb);
                        if ((int)top_candidates.size() > efq)
                            top_candidates.pop();
                        if (!top_candidates.empty())
                            low_bound = top_candidates.top().first;
                    }
                }
            }
        }

        // ( k )
        vector<pair<float, int>> final_candidates;
        while (!top_candidates.empty())
        {
            auto [dist, node] = top_candidates.top();
            top_candidates.pop();
            final_candidates.emplace_back(dist, node);
        }
        sort(final_candidates.begin(), final_candidates.end());
        results.reserve(std::min(k, (int)final_candidates.size()));
        for (const auto &candidate : final_candidates)
        {
            if (base_point_index >= 0 && candidate.second == base_point_index)
            {
                continue;
            }
            results.push_back(candidate.second);
            if ((int)results.size() >= k)
                break;
        }

        return comparison;
    }
};

// ======================== ========================

class PerformanceTester
{
private:
    IndexType &index;
    const DataSetWrapper<float> &base;
    const DataSetWrapper<float> &query;
    const GroundTruth &gt;
    Timer &timer;
    int num_threads = 24;

public:
    PerformanceTester(IndexType &idx,
                      const DataSetWrapper<float> &b,
                      const DataSetWrapper<float> &q,
                      const GroundTruth &ground_truth,
                      Timer &t, int threads = 24)
        : index(idx), base(b), query(q), gt(ground_truth), timer(t), num_threads(threads) {}

    // CSV
    int find_optimal_efq_for_target_recall_tester(IndexType &index,
                                                  const DataSetWrapper<float> &dataset,
                                                  const GroundTruth &gt,
                                                  int k,
                                                  double target_recall,
                                                  int efq_start = 5,
                                                  int coarse_step = 20,
                                                  int refine_step = 2,
                                                  int efq_max = 2000,
                                                  float *achieved_recall = nullptr,
                                                  float *achieved_qps = nullptr)
    {
        const float target = static_cast<float>(target_recall);
        Timer timer;
        index.set_num_threads(24);

        printf(" efq (k=%d, targetrecall=%.4f)...\n", k, target);

        int optimal_efq = -1;
        float optimal_recall = 0.0f;
        float optimal_qps = 0.0f;
        float last_recall = 0.0f;
        float last_qps = 0.0f;

        int lower_efq = -1;
        float lower_recall = 0.0f;
        float lower_qps = 0.0f;
        int upper_efq = -1;
        float upper_recall = 0.0f;
        float upper_qps = 0.0f;

        const int actual_start = std::max(efq_start, k);
        for (int efq = actual_start; efq <= efq_max; efq += coarse_step)
        {
            index.get_comparison_and_clear();
            timer.start();
            const int effective_efq = std::max(efq, k);
            auto [_, knn] = index.search(dataset, k, effective_efq);
            timer.stop();

            float recall = gt.recall(k, knn);
            float qps_val = dataset.num_ / timer.get();
            timer.reset();

            last_recall = recall;
            last_qps = qps_val;

            if (recall >= target)
            {
                upper_efq = efq;
                upper_recall = recall;
                upper_qps = qps_val;
                if (lower_efq == -1)
                {
                    lower_efq = efq - coarse_step;
                    if (lower_efq < actual_start)
                        lower_efq = actual_start;
                }
                break;
            }
            else
            {
                lower_efq = efq;
                lower_recall = recall;
                lower_qps = qps_val;
            }
        }

        if (upper_efq == -1)
        {
            optimal_efq = efq_max;
            optimal_recall = last_recall;
            optimal_qps = last_qps;
            printf(" range target efq, max%d (recall=%.4f, qps=%.1f)\n",
                   optimal_efq, last_recall, last_qps);
        }
        else
        {
            if (lower_efq < actual_start)
            {
                lower_efq = actual_start;
            }
            int refined_upper = upper_efq;
            float refined_recall = upper_recall;
            float refined_qps = upper_qps;

            for (int efq = lower_efq + refine_step; efq < upper_efq; efq += refine_step)
            {
                index.get_comparison_and_clear();
                timer.start();
                const int effective_efq = std::max(efq, k);
                auto [_, knn] = index.search(dataset, k, effective_efq);
                timer.stop();

                float recall = gt.recall(k, knn);
                float qps_val = dataset.num_ / timer.get();
                timer.reset();

                if (recall >= target)
                {
                    refined_upper = efq;
                    refined_recall = recall;
                    refined_qps = qps_val;
                    break;
                }
                else
                {
                    lower_efq = efq;
                    lower_recall = recall;
                    lower_qps = qps_val;
                }
            }

            optimal_efq = refined_upper;
            optimal_recall = refined_recall;
            optimal_qps = refined_qps;
            printf(" range target efq, max%d (recall=%.4f, qps=%.1f)\n",
                   optimal_efq, optimal_recall, optimal_qps, lower_efq, refined_upper);
        }

        if (achieved_recall)
        {
            *achieved_recall = optimal_recall;
        }
        if (achieved_qps)
        {
            *achieved_qps = optimal_qps;
        }

        return optimal_efq;
    }

    void test_and_record(const string &csv_path, int R, int efc, int iter, int k,
                         int truth_added_edges, int tree_added_edges, int cut_edges, int efq_max = 500, int efq_step = 10, double acer_value = -1.0f, double comparison_pers = -1, double opt_time = 0.0)
    {
        vector<float> recalls, qps;

        index.set_num_threads(num_threads);

        // efq
        const int sweep_start = std::max(5, k);
        for (int efq = sweep_start; efq <= efq_max; efq += efq_step)
        {
            index.get_comparison_and_clear();
            timer.start();
            // const int effective_efq = std::max(efq, k);
            auto [_, knn] = index.search(query, k, efq);
            timer.stop();

            float recall = gt.recall(k, knn);
            float qps_val = query.num_ / timer.get();
            
            recalls.emplace_back(recall);
            qps.emplace_back(qps_val);
            timer.reset();
            if(recall>0.995f){
                break;
            }
        }

        // 
        PIGR9Utils::sort_pairs(recalls, qps);
        PIGR9Utils::acquire_frontier(recalls, qps);
        PIGR9Utils::sort_pairs(recalls, qps);

        // CSV
        ofstream csv_file(csv_path, ios::app);
        if (!csv_file.is_open())
        {
            cerr << "Failed to open CSV file: " << csv_path << endl;
            return;
        }

        csv_file << R << "," << efc << "," << iter << "," << k << ","
                 << truth_added_edges << "," << tree_added_edges << "," << cut_edges << ",";

        vector<float> target_recalls = {0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99};
        for (float target_recall : target_recalls)
        {
            float interpolated_qps = PIGR9Utils::linear_interpolate(target_recall, recalls, qps);
            csv_file << fixed << setprecision(3) << interpolated_qps << ",";
        }

        // distance evaluations ACER

        csv_file << comparison_pers << "," << fixed << setprecision(4) << acer_value << "," << fixed << setprecision(2) << opt_time << endl;
        csv_file.close();

        // 
        printf(" range target efq, max%d (recall=%.4f, qps=%.1f)\n",
               iter, k, truth_added_edges + tree_added_edges, cut_edges,
               recalls.empty() ? 0.0f : recalls.front(),
               recalls.empty() ? 0.0f : recalls.back(),
               qps.empty() ? 0.0f : qps.front(),
               qps.empty() ? 0.0f : qps.back());
    }

    // CSV
    void init_csv(const string &csv_path)
    {
        ofstream csv_file(csv_path);
        if (!csv_file.is_open())
        {
            cerr << "Failed to open CSV file: " << csv_path << endl;
            return;
        }

        csv_file << "R,efc,iter,k,truth_add,tree_add,cuteedge,";
        csv_file << "qps(recall=0.6),qps(recall=0.7),qps(recall=0.8),qps(recall=0.9),";
        csv_file << "qps(recall=0.92),qps(recall=0.94),qps(recall=0.96),qps(recall=0.97),";
        csv_file << "qps(recall=0.98),qps(recall=0.99),comp_pers,aecr,opt_time" << endl;
        csv_file.close();

        printf("CSV initialized: %s\n", csv_path.c_str());
    }

    // totaldistance evaluations( )
    // efq : recall >= 0.99 efq , totaldistance evaluations
    pair<double, double> calculate_total_comparisons(int k)
    {
        printf(" totaldistance evaluations, efq (k=%d)...\n", k);

        // 
        printf(" Step 1: efq ( recall >= 0.99 efq )\n");

        // 
        int optimal_efq = -1;
        vector<float> recalls, qps_values;
        Timer timer;
        index.set_num_threads(24);

        printf(" Step 1: efq ( recall >= 0.99 efq )\n");

        // 
        float optimal_recall = 0.0f;
        float optimal_qps = 0.0f;
        float last_recall = 0.0f;
        float last_qps = 0.0f;

        const int scan_start = std::max(5, k);
        for (int efq = scan_start; efq <= 2000; efq += 30)
        {
            index.get_comparison_and_clear();
            timer.start();
            const int effective_efq = std::max(efq, k);
            auto [_, knn] = index.search(query, k, effective_efq);
            timer.stop();
            float recall = gt.recall(k, knn);
            float qps_val = query.num_ / timer.get();
            recalls.push_back(recall);
            qps_values.push_back(qps_val);
            timer.reset();
            last_recall = recall;
            last_qps = qps_val;
            // recall >= 0.99 efq
            if (recall >= 0.99f && optimal_efq == -1)
            {
                optimal_efq = efq;
                optimal_recall = recall;
                optimal_qps = qps_val;
                break;
            }
        }

        if (optimal_efq == -1)
        {
            optimal_efq = 2000;
            printf(" range target efq, max%d (recall=%.4f, qps=%.1f)\n",
                   optimal_efq, last_recall, last_qps);
        }
        else
        {
            printf(" range target efq, max%d (recall=%.4f, qps=%.1f)\n",
                   optimal_efq, optimal_recall, optimal_qps);
        }

        // 2. efq totaldistance evaluations
        printf(" Step 2: compute total distance evaluations on query samples with efq=%d\n", optimal_efq);

        // , 0
        vector<long long> thread_local_comparisons(24, 0LL);
        vector<int> thread_valid_samples(24, 0);
        vector<double> thread_local_acer(24, 0.0);

        printf(" %d , 24 , 0 \n", query.num_);

#pragma omp parallel for num_threads(24) schedule(dynamic, 1000)
        for (int query_idx = 0; query_idx < query.num_; query_idx++)
        {
            int tid = omp_get_thread_num();

            // 
            const float *query_vec = query[query_idx];

            // NSG Branch and Bound
            std::priority_queue<std::pair<float, int>> top_candidates;
            std::priority_queue<std::pair<float, int>> candidate_set;
            std::vector<bool> mass_visited(index.base_.num_, false);
            size_t comparison = 0;

            float init_dist = 0.0f;
            int entry_point = find_start_entry(index, query_vec, init_dist, comparison);
            if (entry_point < 0 || entry_point >= index.base_.num_)
            {
                continue;
            }
            top_candidates.emplace(init_dist, entry_point); // max heap
            candidate_set.emplace(-init_dist, entry_point); // min heap
            mass_visited[entry_point] = true;

            // ( Pruning )
            long long local_adopted = 0;
            long long local_comparisons = 0;

            /// @brief Branch and Bound Algorithm
            float low_bound = init_dist;
            while (candidate_set.size())
            {
                auto curr_el_pair = candidate_set.top();
                if (-curr_el_pair.first > low_bound && top_candidates.size() == optimal_efq)
                    break;
                candidate_set.pop();
                int curr_node_id = curr_el_pair.second;

                const auto *neighbors = get_base_neighbors_const(index, curr_node_id);
                if (!neighbors || neighbors->empty())
                {
                    continue;
                }

                // node
                for (int neighbor_id : *neighbors)
                {
                    if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                        continue;

                    if (!mass_visited[neighbor_id])
                    {
                        mass_visited[neighbor_id] = true;
                        float dd = metrics::euclidean(query_vec, index.base_[neighbor_id], index.base_.dim_);
                        comparison++;
                        local_comparisons++;

                        /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
                        if (top_candidates.top().first > dd || top_candidates.size() < optimal_efq)
                        {
                            candidate_set.emplace(-dd, neighbor_id);
                            top_candidates.emplace(dd, neighbor_id);
                            if (top_candidates.size() > optimal_efq) // give up farest result so far
                                top_candidates.pop();
                            if (top_candidates.size())
                                low_bound = top_candidates.top().first;

                            local_adopted++;
                        }
                        // 
                    }
                }
            }

            // distance evaluations
            long long local_discarded = local_comparisons - local_adopted;
            long long total_attempts = local_adopted + local_discarded;

            if (total_attempts > 0)
            {
                // ACER: adopted / (distance evaluations)
                double local_acer_contribution = static_cast<double>(local_adopted) / (total_attempts);

                // distance evaluations
                thread_local_comparisons[tid] += total_attempts;
                thread_valid_samples[tid]++;
                thread_local_acer[tid] += local_acer_contribution;
            }
        }

        // 
        long long total_comparisons = 0LL;
        int valid_samples = 0;
        double total_acer = 0.0;

        for (int tid = 0; tid < 24; tid++)
        {
            total_comparisons += thread_local_comparisons[tid];
            valid_samples += thread_valid_samples[tid];
            total_acer += thread_local_acer[tid];
        }

        // 3. avgdistance evaluations
        double avg_comparisons = 0;
        if (valid_samples > 0)
        {
            avg_comparisons = total_comparisons / valid_samples;

            printf(" Step 1: efq ( recall >= 0.99 efq )\n");
            printf(" Step 2: compute total distance evaluations on query samples with efq=%d\n", optimal_efq);
            printf(" total : %d ( %d)\n", query.num_, valid_samples);
            printf(" avgdistance evaluations: %lf\n", avg_comparisons);
            printf("  ACER (Approximate Candidate Evaluation Ratio): %.6f\n", total_acer / valid_samples);

            // 4. efq k totaldistance evaluations
            printf(" Step 4: efq=%d k=%d \n", optimal_efq, k);

            // 
            index.get_comparison_and_clear();

            // k
            Timer test_timer;
            test_timer.start();
            const int effective_efq = std::max(optimal_efq, k);
            if (effective_efq != optimal_efq)
            {
                printf(" Note: efq=%d k=%d, efq=%d\n", optimal_efq, k, effective_efq);
            }
            auto [_, knn] = index.search(query, k, effective_efq);
            test_timer.stop();

            // totaldistance evaluations
            long long total_comparisons_k = index.get_comparison_and_clear();
            float recall = gt.recall(k, knn);
            float qps = query.num_ / test_timer.get();

            printf("k=%d (efq=%d):\n", k, effective_efq);
            printf("  recall@%d: %.4f\n", k, recall);
            printf("  QPS: %.1f\n", qps);
            printf(" totaldistance evaluations: %lld\n", total_comparisons_k);
        }
        else
        {
            printf(" Step 1: efq ( recall >= 0.99 efq )\n");
            return {0LL, 0.0};
        }

        return {avg_comparisons, total_acer / valid_samples};
    }
};

// ======================== edge ========================

class OptimizedEdgePruner
{
private:
    IndexType &index;
    int num_threads = 24;

public:
    OptimizedEdgePruner(IndexType &idx, int threads = 24) : index(idx), num_threads(threads) {}

    // edge - ( PIGR9_clear , )
    MultiLevelEdgeStats
    build_edge_statistics_optimized(const DataSetWrapper<float> &base, int efq_maintree, int total_batches,
                                    const vector<int> *sampled_indices = nullptr,
                                    int level_limit = std::numeric_limits<int>::max())
    {
        printf(" Step 1: efq ( recall >= 0.99 efq )\n");
        Timer t;
        t.start();

        const vector<int> *indices = sampled_indices;
        const int total_points = indices ? static_cast<int>(indices->size()) : static_cast<int>(base.num_);
        if (total_points <= 0)
        {
            printf(" Step 1: efq ( recall >= 0.99 efq )\n");
            return MultiLevelEdgeStats(index.link_lists_.size());
        }

        printf(" minefq=%d (recall=%.4f, qps=%.1f), [%d, %d]\n",
               base.num_, total_points, base.num_ > 0 ? (100.0 * total_points / base.num_) : 0.0);

        const int effective_level_limit = (level_limit <= 0) ? 1 : level_limit;
        if (level_limit != std::numeric_limits<int>::max())
        {
            printf(" levelrange: 0~%d\n", effective_level_limit - 1);
        }

        // edge_stats_sparse[node_id][neighbor_id] = {adopted, discarded}
        // edge_stats_sparse[node_id][neighbor_id] = {adopted, discarded}
        unordered_map<EdgeKey, pair<int, int>, EdgeKeyHash> edge_stats_sparse;

        printf(" Step 1: efq ( recall >= 0.99 efq )\n");

        // total parameters
        const int batch_size = max(1, (total_points + total_batches - 1) / total_batches);

        printf(" %d , %d node ( )\n", total_batches, batch_size);

        for (int batch = 0; batch < total_batches; batch++)
        {
            int start_idx = batch * batch_size;
            if (start_idx >= total_points)
            {
                break;
            }
            int end_idx = min(total_points, start_idx + batch_size);

            printf(" %d/%d : range[%d, %d)\n", batch + 1, total_batches, start_idx, end_idx);

            // , local
            vector<unordered_map<EdgeKey, pair<int, int>, EdgeKeyHash>> batch_locals(this->num_threads);

#pragma omp parallel for schedule(dynamic, 32) num_threads(this->num_threads)
            for (int global_idx = start_idx; global_idx < end_idx; global_idx++)
            {
                int start_id = indices ? (*indices)[global_idx] : global_idx;
                if (start_id < 0 || start_id >= base.num_)
                {
                    continue;
                }
                int tid = omp_get_thread_num();

                // :
                if (index.enterpoint_node_ < 0 || index.enterpoint_node_ >= index.base_.num_)
                {
                    continue;
                }

                // NSG Branch and Bound
                std::priority_queue<std::pair<float, int>> top_candidates;
                std::priority_queue<std::pair<float, int>> candidate_set;
                std::vector<bool> mass_visited(index.base_.num_, false);
                size_t comparison = 0;

                float init_dist = 0.0f;
                std::vector<std::tuple<int, int, int, bool>> level_events;
                level_events.reserve(64);

                int entry_point = find_start_entry(index, base[start_id], init_dist, comparison, nullptr, -1, &level_events);
                if (entry_point < 0 || entry_point >= index.base_.num_)
                {
                    continue;
                }
                top_candidates.emplace(init_dist, entry_point); // max heap
                candidate_set.emplace(-init_dist, entry_point); // min heap
                mass_visited[entry_point] = true;

                auto &thread_map = batch_locals[tid];

                for (const auto &evt : level_events)
                {
                    int from = std::get<0>(evt);
                    int to = std::get<1>(evt);
                    int level = std::get<2>(evt);
                    if (level < 0 || level >= effective_level_limit)
                    {
                        continue;
                    }
                    bool adopted = std::get<3>(evt);
                    if (from < 0 || from >= index.base_.num_ || to < 0 || to >= index.base_.num_)
                    {
                        continue;
                    }
                    EdgeKey key(from, to, level);
                    auto &stat = thread_map[key];
                    if (adopted)
                    {
                        stat.first++;
                    }
                    else
                    {
                        stat.second++;
                    }
                }

                // 
                int total_edge_checks = 0;
                int adopted_edges = 0;
                int discarded_edges = 0;

                /// @brief Branch and Bound Algorithm
                float low_bound = init_dist;
                while (candidate_set.size())
                {
                    auto curr_el_pair = candidate_set.top();
                    if (-curr_el_pair.first > low_bound && top_candidates.size() == efq_maintree)
                        break;
                    candidate_set.pop();
                    int curr_node_id = curr_el_pair.second;

                    const auto *neighbors = get_base_neighbors_const(index, curr_node_id);
                    if (!neighbors || neighbors->empty())
                    {
                        continue;
                    }

                    // node
                    for (int neighbor_id : *neighbors)
                    {
                        if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                            continue;

                        if (!mass_visited[neighbor_id])
                        {
                            mass_visited[neighbor_id] = true;
                            float dist = metrics::euclidean(base[start_id], index.base_[neighbor_id], index.base_.dim_);
                            total_edge_checks++;

                            EdgeKey base_key(curr_node_id, neighbor_id, 0);
                            auto &stat = thread_map[base_key];

                            // : edge (curr_node_id, neighbor_id),
                            if (top_candidates.top().first > dist || top_candidates.size() < efq_maintree)
                            {
                                candidate_set.emplace(-dist, neighbor_id);
                                top_candidates.emplace(dist, neighbor_id);
                                if (top_candidates.size() > efq_maintree) // give up farest result so far
                                    top_candidates.pop();
                                if (top_candidates.size())
                                    low_bound = top_candidates.top().first;

                                stat.first++;
                                adopted_edges++;
                            }
                            else
                            {
                                // 
                                stat.second++;
                                discarded_edges++;
                            }
                        }
                    }
                }

                // ( 5 node )
                if ((global_idx % 50000) == 0)
                {
                    // 
                    printf(" efq scan: first efq achieving recall>=0.99 is %d (recall=%.4f, qps=%.1f)\n",
                           start_id, total_edge_checks, adopted_edges, discarded_edges, efq_maintree);
                }
            }

            // global
            printf(" %d ...\n", batch + 1);
            long long total_adopted_in_batch = 0;
            long long total_discarded_in_batch = 0;

            for (int tid = 0; tid < this->num_threads; tid++)
            {
                for (const auto &entry : batch_locals[tid])
                {
                    const EdgeKey &key = entry.first;
                    const auto &stat = entry.second;
                    if (stat.first > 0 || stat.second > 0)
                    {
                        auto &global_stat = edge_stats_sparse[key];
                        global_stat.first += stat.first;
                        global_stat.second += stat.second;
                        total_adopted_in_batch += stat.first;
                        total_discarded_in_batch += stat.second;
                    }
                }
            }

            printf(" efq scan: first efq achieving recall>=0.99 is %d (recall=%.4f, qps=%.1f)\n",
                   batch + 1, total_adopted_in_batch, total_discarded_in_batch);

            // local ,
            for (auto &local_map : batch_locals)
            {
                unordered_map<EdgeKey, pair<int, int>, EdgeKeyHash>().swap(local_map);
            }
        }

        t.stop();
        printf("edge , : %.2f \n", t.get());

        // ( Pruning )
        const size_t node_count = index.link_lists_.size();
        MultiLevelEdgeStats edge_stats_by_node(node_count);
        for (size_t node = 0; node < node_count; ++node)
        {
            const auto &levels = index.link_lists_[node];
            const size_t level_count = std::min(levels.size(), static_cast<size_t>(effective_level_limit));
            edge_stats_by_node[node].resize(level_count);
            for (size_t level = 0; level < level_count; ++level)
            {
                edge_stats_by_node[node][level].assign(levels[level].size(), {0, 0});
            }
        }

        for (const auto &entry : edge_stats_sparse)
        {
            const EdgeKey &key = entry.first;
            const auto &stat = entry.second;
            if (key.from < 0 || key.from >= static_cast<int>(node_count))
            {
                continue;
            }
            if (key.level < 0 || key.level >= static_cast<int>(edge_stats_by_node[key.from].size()))
            {
                continue;
            }
            if (key.level >= static_cast<int>(index.link_lists_[key.from].size()))
            {
                continue;
            }
            auto &neighbors = index.link_lists_[key.from][key.level];
            auto &stats_vec = edge_stats_by_node[key.from][key.level];
            auto it = std::find(neighbors.begin(), neighbors.end(), key.to);
            if (it == neighbors.end())
            {
                continue;
            }
            size_t idx = static_cast<size_t>(std::distance(neighbors.begin(), it));
            if (idx >= stats_vec.size())
            {
                continue;
            }
            stats_vec[idx].first += stat.first;
            stats_vec[idx].second += stat.second;
        }

        // :
        int total_edges_sparse = static_cast<int>(edge_stats_sparse.size());
        long long total_adopted_sparse = 0;
        long long total_discarded_sparse = 0;
        for (const auto &entry : edge_stats_sparse)
        {
            total_adopted_sparse += entry.second.first;
            total_discarded_sparse += entry.second.second;
        }

        int total_edges_converted = 0;
        long long total_adopted_converted = 0;
        long long total_discarded_converted = 0;
        for (const auto &node_levels : edge_stats_by_node)
        {
            for (const auto &level_stats : node_levels)
            {
                for (const auto &stat : level_stats)
                {
                    if (stat.first > 0 || stat.second > 0)
                    {
                        total_edges_converted++;
                        total_adopted_converted += stat.first;
                        total_discarded_converted += stat.second;
                    }
                }
            }
        }

        printf(" edge ( )...\n");
        printf(" : %zu, : %d ( =%.2f%%)\n",
               total_edges_sparse, total_adopted_sparse, total_discarded_sparse);
        printf(" : %zu, : %d ( =%.2f%%)\n",
               total_edges_converted, total_adopted_converted, total_discarded_converted);

        if (total_edges_sparse != total_edges_converted ||
            total_adopted_sparse != total_adopted_converted ||
            total_discarded_sparse != total_discarded_converted)
        {
            printf("Warning: <=0, skip . \n");
        }
        else
        {
            printf("Warning: <=0, skip . \n");
        }

        printf("Warning: <=0, skip . \n");
        printf("Warning: <=0, skip . \n");

        // 
        {
            decltype(edge_stats_sparse)().swap(edge_stats_sparse);
            trim_os_memory_all_threads();
        }

        return edge_stats_by_node;
    }

    // Pruning
    int prune_edges_by_statistics_optimized(const vector<vector<pair<int, int>>> &edge_stats_by_node,
                                            float prune_ratio)
    {
        printf(" , ...\n");
        printf(" Pruning : %.1f%%\n", prune_ratio * 100);

        // discarded/adopted : adopted/adopted edge
        struct EdgeStat
        {
            int node;
            int neighbor;
            int adopted;
            int discarded;
            int total;
            double discard_adopt_ratio;
        };
        std::vector<EdgeStat> all_edges;
        std::vector<double> discard_adopt_ratios;
        discard_adopt_ratios.reserve(edge_stats_by_node.size());
        int total_valid_edges = 0, edges_with_stats = 0, no_stats_count = 0;
        int adopted_zero_total = 0, adopted_zero_with_discard = 0;
        for (int node = 0; node < (int)edge_stats_by_node.size(); node++)
        {
            if (node < 0 || node >= static_cast<int>(index.link_lists_.size()))
            {
                continue;
            }
            const auto *neighbors = get_base_neighbors_const(index, node);
            int neighbor_count = (neighbors != nullptr)
                                     ? std::min(static_cast<int>(edge_stats_by_node[node].size()), static_cast<int>(neighbors->size()))
                                     : 0;
            for (int neighbor_idx = 0; neighbor_idx < neighbor_count; neighbor_idx++)
            {
                const auto &stat = edge_stats_by_node[node][neighbor_idx];
                total_valid_edges++;
                int adopted = stat.first;
                int discarded = stat.second;
                int total = adopted + discarded;
                if (total == 0)
                {
                    no_stats_count++;
                    continue;
                }
                int neighbor_id = (*neighbors)[neighbor_idx];
                if (neighbor_id >= 0 && neighbor_id < index.base_.num_)
                {
                    if (adopted == 0)
                    {
                        adopted_zero_total++;
                        if (discarded > 0)
                        {
                            adopted_zero_with_discard++;
                        }
                        continue;
                    }
                    double ratio = static_cast<double>(discarded) / static_cast<double>(adopted);
                    EdgeStat es{node, neighbor_id, adopted, discarded, total, ratio};
                    all_edges.push_back(es);
                    discard_adopt_ratios.push_back(ratio);
                    edges_with_stats++;
                }
            }
        }
        double avg_ratio = 0.0, median_ratio = 0.0, q1_ratio = 0.0, q3_ratio = 0.0;
        double p90_ratio = 0.0, p95_ratio = 0.0, p99_ratio = 0.0;
        if (!discard_adopt_ratios.empty())
        {
            std::sort(discard_adopt_ratios.begin(), discard_adopt_ratios.end());
            avg_ratio = std::accumulate(discard_adopt_ratios.begin(), discard_adopt_ratios.end(), 0.0) / discard_adopt_ratios.size();
            auto percentile = [&](double p) -> double
            {
                if (discard_adopt_ratios.empty())
                    return 0.0;
                double idx = (discard_adopt_ratios.size() - 1) * p;
                size_t lo = static_cast<size_t>(std::floor(idx));
                size_t hi = static_cast<size_t>(std::ceil(idx));
                double frac = idx - lo;
                double val_lo = discard_adopt_ratios[lo];
                double val_hi = discard_adopt_ratios[hi];
                return val_lo + (val_hi - val_lo) * frac;
            };
            q1_ratio = percentile(0.25);
            median_ratio = percentile(0.50);
            q3_ratio = percentile(0.75);
            p90_ratio = percentile(0.90);
            p95_ratio = percentile(0.95);
            p99_ratio = percentile(0.99);
            printf(" Node %d: checked %d edges, adopted %d, discarded %d, final candidate set size %d (pruning parameters)\n",
                   avg_ratio, median_ratio, q1_ratio, q3_ratio);
        }
        printf(" Node %d: checked %d edges, adopted %d, discarded %d, final candidate set size %d (pruning parameters)\n",
               all_edges.size(), adopted_zero_total, adopted_zero_with_discard);
        if (all_edges.empty())
        {
            if (adopted_zero_with_discard > 0)
            {
                printf(" , ...\n");
            }
            else
            {
                printf(" , ...\n");
            }
        }
        std::sort(all_edges.begin(), all_edges.end(), [](const EdgeStat &a, const EdgeStat &b)
                  {
            if (std::fabs(a.discard_adopt_ratio - b.discard_adopt_ratio) > 1e-12) {
                return a.discard_adopt_ratio > b.discard_adopt_ratio;
            }
            if (a.total != b.total) {
                return a.total > b.total;
            }
            return a.discarded > b.discarded; });
        size_t target_num = static_cast<size_t>(all_edges.size() * prune_ratio);
        if (target_num > all_edges.size())
            target_num = all_edges.size();
        std::set<std::pair<int, int>> high_ratio_set;
        for (size_t i = 0; i < target_num; ++i)
        {
            high_ratio_set.insert({all_edges[i].node, all_edges[i].neighbor});
        }
        if (!all_edges.empty())
        {
            double max_ratio = all_edges.front().discard_adopt_ratio;
            double min_ratio = all_edges.back().discard_adopt_ratio;
            double threshold = (target_num > 0) ? all_edges[target_num - 1].discard_adopt_ratio : 0.0;
            printf(" Node %d: checked %d edges, adopted %d, discarded %d, final candidate set size %d (pruning parameters)\n",
                   max_ratio, median_ratio, p90_ratio, p95_ratio, p99_ratio, min_ratio);
            printf(" Node %d: checked %d edges, adopted %d, discarded %d, final candidate set size %d (pruning parameters)\n",
                   prune_ratio * 100, target_num, target_num, threshold);

            size_t topn = std::min<size_t>(10, all_edges.size());
            printf("Top%d adopted/adopted edge(global):\n", (int)topn);
            for (size_t i = 0; i < topn; ++i)
            {
                const char *flag = (i < target_num ? " [selected]" : "");
                printf("  #%zu: %d->%d, ratio=%.4f, visits=%d, adopted=%d, discarded=%d%s\n", i + 1,
                       all_edges[i].node, all_edges[i].neighbor, all_edges[i].discard_adopt_ratio,
                       all_edges[i].total, all_edges[i].adopted, all_edges[i].discarded, flag);
            }
        }
        printf(" adopted/adopted %.1f%%edge, =%zu\n", prune_ratio * 100, target_num);
        if (!all_edges.empty() && target_num > 0)
        {
            printf(" :\n");
            size_t show_n = std::min<size_t>(10, target_num);
            for (size_t i = 0; i < show_n; ++i)
            {
                printf("  #%zu: %d->%d, ratio=%.4f, visits=%d, adopted=%d, discarded=%d\n", i + 1,
                       all_edges[i].node, all_edges[i].neighbor, all_edges[i].discard_adopt_ratio,
                       all_edges[i].total, all_edges[i].adopted, all_edges[i].discarded);
            }
        }

        // remove edge
        vector<pair<int, int>> edges_to_delete;
        vector<bool> node_cut_flag(index.base_.num_, false);
        int node_limit_blocked = 0;
        int marked_redundant = 0;
        int marked_notkept = 0;
        int max_nodes = std::min(static_cast<int>(edge_stats_by_node.size()), static_cast<int>(index.link_lists_.size()));
        for (int node = 0; node < max_nodes; node++)
        {
            if (node < 0 || node >= static_cast<int>(edge_stats_by_node.size()))
            {
                continue;
            }
            const auto *neighbors = get_base_neighbors_const(index, node);
            if (!neighbors)
            {
                continue;
            }
            int neighbor_count = std::min(static_cast<int>(edge_stats_by_node[node].size()), static_cast<int>(neighbors->size()));
            for (int neighbor_idx = 0; neighbor_idx < neighbor_count; neighbor_idx++)
            {
                const auto &stat = edge_stats_by_node[node][neighbor_idx];
                int adopted_count = stat.first;
                int discarded_count = stat.second;
                int neighbor_id = (*neighbors)[neighbor_idx];
                if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                {
                    continue;
                }
                // skip edge
                if (adopted_count == 0 && discarded_count == 0)
                {
                    continue;
                }
                // edge remove(adopted==0 discarded)
                if (adopted_count == 0 && discarded_count > 0)
                {
                    if (!node_cut_flag[node])
                    {
                        edges_to_delete.emplace_back(node, neighbor_id);
                        node_cut_flag[node] = true;
                        marked_redundant++;
                    }
                    else
                    {
                        node_limit_blocked++;
                    }
                    continue;
                }
                // edge, remove
                if (adopted_count > 0 && high_ratio_set.count({node, neighbor_id}) > 0)
                {
                    if (!node_cut_flag[node])
                    {
                        edges_to_delete.emplace_back(node, neighbor_id);
                        node_cut_flag[node] = true;
                        marked_notkept++;
                    }
                    else
                    {
                        node_limit_blocked++;
                    }
                }
            }
        }
        // node , adopted
        int cut_count = 0;
        std::map<int, std::vector<std::pair<int, double>>> node2del;
        std::map<std::pair<int, int>, EdgeStat> edge2info;
        for (const auto &edge : edges_to_delete)
        {
            int node = edge.first, neighbor = edge.second;
            edge2info[{node, neighbor}] = EdgeStat{node, neighbor, 0, 0, 0, 0.0};
        }
        for (const auto &e : all_edges)
        {
            if (edge2info.count({e.node, e.neighbor}))
            {
                edge2info[{e.node, e.neighbor}] = e;
            }
        }
        for (const auto &edge : edges_to_delete)
        {
            int node = edge.first, neighbor = edge.second;
            const auto &info = edge2info[{node, neighbor}];
            double ratio = info.total > 0 ? info.discard_adopt_ratio : std::numeric_limits<double>::infinity();
            node2del[node].emplace_back(neighbor, ratio);
        }
        std::map<int, int> node_total_visits;
        for (const auto &e : all_edges)
        {
            node_total_visits[e.node] += e.total;
        }
        std::vector<std::pair<int, int>> final_del;
        for (const auto &kv : node2del)
        {
            int node = kv.first;
            const auto &del_list = kv.second;
            double best_ratio = -1.0;
            int best_neighbor = -1;
            EdgeStat best_info{node, -1, 0, 0, 0, 0.0};
            for (const auto &[neighbor, ratio] : del_list)
            {
                if (ratio > best_ratio)
                {
                    best_ratio = ratio;
                    best_neighbor = neighbor;
                    best_info = edge2info[{node, neighbor}];
                }
            }
            if (best_neighbor != -1)
            {
                final_del.emplace_back(node, best_neighbor);
                int total_visits = node_total_visits[node];
                printf(" : edge =%d, adopted=%lld, discarded=%lld\n",
                       node, total_visits, node, best_neighbor, best_info.total,
                       best_info.discarded, best_info.discard_adopt_ratio);
            }
        }
        for (const auto &edge : edges_to_delete)
        {
            int node = edge.first, neighbor = edge.second;
            const auto &info = edge2info[{node, neighbor}];
            if (info.adopted == 0 && info.discarded > 0)
            {
                final_del.emplace_back(node, neighbor);
            }
        }
        for (const auto &edge : final_del)
        {
            int node1 = edge.first;
            int node2 = edge.second;
            if (node1 < 0 || node1 >= static_cast<int>(index.link_lists_.size()) ||
                node2 < 0 || node2 >= index.base_.num_)
            {
                continue;
            }
            auto *neighbors = get_base_neighbors_mut(index, node1);
            if (!neighbors)
            {
                continue;
            }
            auto it = remove(neighbors->begin(), neighbors->end(), node2);
            if (it != neighbors->end())
            {
                neighbors->erase(it, neighbors->end());
                cut_count++;
            }
        }
        printf(" edge ( )...\n");
        printf(" totaledge : %d\n", total_valid_edges);
        printf(" Edges with stats and adopted>0: %d\n", edges_with_stats);
        printf(" Edges with adopted==0: %d (incl. discarded>0=%d)\n", adopted_zero_total, adopted_zero_with_discard);
        printf(" Edges without stats: %d (skipped)\n", no_stats_count);
        printf(" Redundant edges removed directly: %d\n", marked_redundant);
        printf(" Edges removed because not in keep-set: %d\n", marked_notkept);
        printf(" adopted %.1f%%edge, removetotal : %d\n", prune_ratio * 100, (int)edges_to_delete.size());
        printf(" Removals blocked by node constraints: %d\n", node_limit_blocked);
        printf(" remove edge : %d\n", cut_count);
        int show_count = min(10, (int)edges_to_delete.size());
        printf(" : candidateedge adopted==0, edge remove. \n");
        for (int i = 0; i < show_count; i++)
        {
            printf(" edge%d: %d->%d\n", i, edges_to_delete[i].first, edges_to_delete[i].second);
        }
        printf(" : candidateedge adopted==0, edge remove. \n");
        return cut_count;
    }
};

// ======================== Edge insertion ( ) ========================

class SmartEdgeAdder
{
private:
    IndexType &index;
    int num_threads = 24;
    int return_k_ = 10;

    inline size_t matrix_index(int r, int c) const
    {
        return static_cast<size_t>(r) * static_cast<size_t>(return_k_) + static_cast<size_t>(c);
    }

    inline void set_matrix_value(std::vector<int> &matrix, int r, int c, int value) const
    {
        if (return_k_ <= 0)
        {
            return;
        }
        const size_t idx = matrix_index(r, c);
        if (idx < matrix.size())
        {
            matrix[idx] = value;
        }
    }

    inline int get_matrix_value(const std::vector<int> &matrix, int r, int c) const
    {
        if (return_k_ <= 0)
        {
            return 0;
        }
        const size_t idx = matrix_index(r, c);
        return idx < matrix.size() ? matrix[idx] : 0;
    }

public:
    SmartEdgeAdder(IndexType &idx, int threads = 24, int return_k = 10)
        : index(idx), num_threads(threads), return_k_(return_k) {}

    void set_return_k(int k)
    {
        return_k_ = k;
    }

    // Edge insertion - PIGR9_clear
    int process_single_tree_and_add_edges(const float *query_data, int start_id,
                                          int efq_maintree, int jump_max, int repeat_max,
                                          int level)
    {
        int add_count = 0;
        if (return_k_ <= 0)
        {
            return 0;
        }

        // : start_id
        if (start_id < 0 || start_id >= index.base_.num_)
        {
            return 0;
        }

        if (level < 0)
        {
            return 0;
        }
        int node_level = get_node_level(index, start_id);
        if (node_level < level)
        {
            return 0;
        }

        const auto *start_level_neighbors = get_neighbors_const(index, start_id, level);
        if (!start_level_neighbors)
        {
            return 0;
        }

        // 1. ( PIGR9_clear )
        std::priority_queue<std::pair<float, int>> visited;
        std::unordered_set<int> vis;

        // 
        float init_dist = 0.0f;
        visited.push(std::make_pair(-init_dist, start_id));
        vis.insert(start_id);

        // node ( )
        std::vector<std::pair<int, float>> node_dist_list;
        node_dist_list.emplace_back(start_id, init_dist);

        // 2. - PIGR9_clear
        int search_iterations = 0;
        const int max_search_iterations = efq_maintree * 2;

        while (!visited.empty() && vis.size() < efq_maintree && search_iterations < max_search_iterations)
        {
            auto top = visited.top();
            visited.pop();
            float neg_dist = top.first;
            int node_id = top.second;
            search_iterations++;

            // : nodeID
            if (node_id < 0 || node_id >= index.base_.num_)
            {
                continue;
            }

            // : node
            const auto *node_neighbors = get_base_neighbors_const(index, node_id);
            if (!node_neighbors || node_neighbors->empty())
            {
                continue;
            }

            // -
            std::vector<int> neighbors_copy;
#pragma omp critical(neighbor_read)
            {
                // 
                const auto *neighbors_snapshot = get_base_neighbors_const(index, node_id);
                if (neighbors_snapshot && !neighbors_snapshot->empty())
                {
                    neighbors_copy = *neighbors_snapshot;
                }
            }

            // 
            for (int neighbor : neighbors_copy)
            {
                // : ID
                if (neighbor < 0 || neighbor >= index.base_.num_)
                {
                    continue;
                }

                if (vis.find(neighbor) == vis.end())
                {
                    vis.insert(neighbor);
                    float dist = metrics::euclidean(query_data, index.base_[neighbor], index.base_.dim_);
                    visited.push(std::make_pair(-dist, neighbor));

                    // node
                    node_dist_list.emplace_back(neighbor, dist);
                }
            }
        }

        // :
        if (vis.size() < 3)
        {
            return 0;
        }

        // 3. Edge insertion - PIGR9_clear
        add_count += process_tree_for_edges(node_dist_list, start_id, jump_max, repeat_max, level);

        return add_count;
    }

    // edge - PIGR9_clear
    int process_tree_for_edges(const std::vector<std::pair<int, float>> &node_dist_list,
                               int tree_id, int jump_max, int repeat_max, int level)
    {
        int add_count = 0;
        if (return_k_ <= 0)
        {
            return 0;
        }

        // : node
        if (node_dist_list.size() < 2)
        {
            return 0;
        }

        // 1. , min k_subadj node - PIGR9_clear
        std::vector<std::pair<int, float>> sorted_nodes = node_dist_list;
        std::sort(sorted_nodes.begin(), sorted_nodes.end(),
                  [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                  { return a.second < b.second; });

        int num = std::min(static_cast<int>(sorted_nodes.size()), return_k_);
        if (sorted_nodes.size() > num)
            sorted_nodes.resize(num);

        // targetlevel node
        std::vector<std::pair<int, float>> filtered_nodes;
        filtered_nodes.reserve(sorted_nodes.size());
        for (const auto &entry : sorted_nodes)
        {
            if (get_node_level(index, entry.first) >= level)
            {
                filtered_nodes.push_back(entry);
            }
        }

        if (filtered_nodes.size() < 2)
        {
            return 0;
        }

        if (filtered_nodes.size() > static_cast<size_t>(return_k_))
        {
            filtered_nodes.resize(return_k_);
        }

        sorted_nodes.swap(filtered_nodes);
        num = static_cast<int>(sorted_nodes.size());

        // 2. node id, id - PIGR9_clear
        std::vector<int> selected_ids;
        std::unordered_map<int, int> id2idx;
        for (int i = 0; i < sorted_nodes.size(); ++i)
        {
            selected_ids.emplace_back(sorted_nodes[i].first);
            id2idx[sorted_nodes[i].first] = i;
        }

        // target (tree_id) selected_ids
        bool target_in_selected = false;
        int target_index = -1;
        for (int i = 0; i < selected_ids.size(); ++i)
        {
            if (selected_ids[i] == tree_id)
            {
                target_in_selected = true;
                target_index = i;
                break;
            }
        }

        if (!target_in_selected)
        {
            // target node , target node
            int replace_index = selected_ids.size() - 1;
            selected_ids[replace_index] = tree_id;
            id2idx[tree_id] = replace_index;

            // node_dist_list
            // tree_id node_dist_list
            float target_dist = 0.0f;
            for (const auto &pair : node_dist_list)
            {
                if (pair.first == tree_id)
                {
                    target_dist = pair.second;
                    break;
                }
            }
            sorted_nodes[replace_index] = {tree_id, target_dist};
        }

        // 3. A0 - PIGR9_clear
        std::vector<int> A0(static_cast<size_t>(return_k_) * static_cast<size_t>(return_k_), 0);

        for (int i = 0; i < selected_ids.size(); ++i)
        {
            int u = selected_ids[i];

            // : nodeID
            const auto *neighbors_u = get_neighbors_const(index, u, level);
            if (u < 0 || u >= index.base_.num_ || !neighbors_u || neighbors_u->empty())
            {
                continue;
            }

            // -
            std::vector<int> neighbors_copy;
#pragma omp critical(neighbor_read)
            {
                // nodeID
                const auto *neighbors_snapshot = get_neighbors_const(index, u, level);
                if (neighbors_snapshot && !neighbors_snapshot->empty())
                {
                    neighbors_copy = *neighbors_snapshot;
                }
            }

            for (int v : neighbors_copy)
            {
                // : ID
                if (v < 0 || v >= index.base_.num_)
                {
                    continue;
                }

                auto it = id2idx.find(v);
                if (it != id2idx.end())
                {
                    int j = it->second;
                    if (j >= 0 && j < num)
                    {
                        set_matrix_value(A0, i, j, 1);
                    }
                }
            }
        }

        // 4. edge - PIGR9_clear
        for (int repeat = 0; repeat < repeat_max; repeat++)
        {
            // Am1 = A0^jump_max ( )
            std::vector<int> Am1 = A0;

            for (int jump = 1; jump < jump_max; ++jump)
            {
                std::vector<int> temp(static_cast<size_t>(return_k_) * static_cast<size_t>(return_k_), 0);
                for (int i = 0; i < num; ++i)
                {
                    for (int j = 0; j < num; ++j)
                    {
                        for (int k = 0; k < num; ++k)
                        {
                            if (get_matrix_value(Am1, i, j) == 1 ||
                                (get_matrix_value(Am1, i, k) && get_matrix_value(A0, k, j)))
                            {
                                set_matrix_value(temp, i, j, 1);
                                break;
                            }
                        }
                    }
                }
                Am1.swap(temp);
            }

            // 0
            int max_zeros = 0;
            int best_row = -1;
            for (int i = 0; i < num; ++i)
            {
                int zero_count = 0;
                for (int j = 0; j < num; ++j)
                {
                    if (get_matrix_value(Am1, i, j) == 0)
                        zero_count++;
                }
                if (zero_count > max_zeros)
                {
                    max_zeros = zero_count;
                    best_row = i;
                }
            }

            if (max_zeros == 0 || best_row == -1)
                break;

            // Am2 = A0^(jump_max-1)
            std::vector<int> Am2 = A0;
            for (int jump = 1; jump < jump_max - 1; ++jump)
            {
                std::vector<int> temp(static_cast<size_t>(return_k_) * static_cast<size_t>(return_k_), 0);
                for (int i = 0; i < num; ++i)
                {
                    for (int j = 0; j < num; ++j)
                    {
                        for (int k = 0; k < num; ++k)
                        {
                            if (get_matrix_value(Am2, i, j) == 1 ||
                                (get_matrix_value(Am2, i, k) && get_matrix_value(A0, k, j)))
                            {
                                set_matrix_value(temp, i, j, 1);
                                break;
                            }
                        }
                    }
                }
                Am2.swap(temp);
            }

            // edge candidate
            int best_col = -1;
            int max_connections = 0;
            for (int j = 0; j < num; ++j)
            {
                if (get_matrix_value(Am1, best_row, j) == 0)
                {
                    int connection_count = 0;
                    for (int i = 0; i < num; ++i)
                    {
                        if (get_matrix_value(Am2, j, i) == 1)
                            connection_count++;
                    }
                    if (connection_count > max_connections)
                    {
                        max_connections = connection_count;
                        best_col = j;
                    }
                }
            }

            // Edge insertion: best_row -> best_col -
            if (best_col != -1 && best_row >= 0 && best_row < num && best_col >= 0 && best_col < num)
            {
                int u = selected_ids[best_row];
                int v = selected_ids[best_col];

                // : nodeID
                if (u >= 0 && u < index.base_.num_ && v >= 0 && v < index.base_.num_ && u != v)
                {
// , global
#pragma omp critical(edge_addition)
                    {
                        // critical section ,
                        auto *neighbors_u_ptr = get_neighbors_mut(index, u, level);
                        auto *neighbors_v_ptr = get_neighbors_mut(index, v, level);
                        if (neighbors_u_ptr && neighbors_v_ptr)
                        {
                            auto &neighbors_u = *neighbors_u_ptr;
                            auto &neighbors_v = *neighbors_v_ptr;

                            // edge
                            bool edge_u_to_v = std::find(neighbors_u.begin(), neighbors_u.end(), v) != neighbors_u.end();
                            bool edge_v_to_u = std::find(neighbors_v.begin(), neighbors_v.end(), u) != neighbors_v.end();

                            // edge( )
                            bool added_u_to_v = false;
                            bool added_v_to_u = false;

                            if (!edge_u_to_v)
                            {
                                neighbors_u.emplace_back(v);
                                added_u_to_v = true;
                                add_count++;
                            }
                            if (!edge_v_to_u)
                            {
                                neighbors_v.emplace_back(u);
                                added_v_to_u = true;
                                add_count++;
                            }
                        }
                    }
                    // critical section , A0 local
                    if (best_row < return_k_ && best_col < return_k_)
                    {
                        set_matrix_value(A0, best_row, best_col, 1);
                        set_matrix_value(A0, best_col, best_row, 1);
                    }
                }
            }
        }

        return add_count;
    }

    // Edge insertion - PIGR9_clear
    int add_edges_by_search_tree(const DataSetWrapper<float> &base,
                                 int efq_maintree, int jump_max = 5, int repeat_max = 5, int return_k = 10,
                                 int allowed_levels = 1)
    {
        printf(" Edge insertion ( )...\n");
        Timer t;
        t.start();
        if (return_k > 0 && return_k != return_k_)
        {
            set_return_k(return_k);
        }
        if (return_k_ <= 0)
        {
            printf(" Edge insertion ( )...\n");
            return 0;
        }

        const int level_limit = std::max(1, allowed_levels);
        if (level_limit == 1)
        {
            printf(" Edge insertion ( )...\n");
        }
        else
        {
            printf("localEdge insertionlevel : %d level(0~%d). \n", level_limit, level_limit - 1);
        }

        printf(" : %zu, %d \n", base.num_, num_threads);
        printf(" Edge insertion ( )...\n");

        // :
        int disconnected_nodes = 0;
        int total_edges = 0;
        for (int i = 0; i < index.base_.num_; i++)
        {
            const auto *neighbors = get_base_neighbors_const(index, i);
            int degree = neighbors ? static_cast<int>(neighbors->size()) : 0;
            total_edges += degree;
            if (degree == 0)
            {
                disconnected_nodes++;
            }
        }

        printf(" : totaledge =%d, node =%d (%.2f%%)\n",
               total_edges, disconnected_nodes, 100.0 * disconnected_nodes / index.base_.num_);

        if (disconnected_nodes > index.base_.num_ * 0.1)
        {
            printf(" : totaledge =%d, node =%d (%.2f%%)\n",
                   100.0 * disconnected_nodes / index.base_.num_);
            return 0;
        }

        printf(" Edge insertion ( )...\n");

        // node-level ( level )
        std::vector<std::pair<int, int>> node_level_tasks;
        node_level_tasks.reserve(base.num_ * std::min(level_limit, 2));

        const int max_global_level = std::max(0, get_max_level(index));
        for (int node = 0; node < index.base_.num_; ++node)
        {
            const auto level_count = static_cast<int>(index.link_lists_[node].size());
            if (level_count <= 0)
            {
                continue;
            }
            for (int level = level_count - 1; level >= 0; --level)
            {
                if (level >= level_limit)
                {
                    continue;
                }
                node_level_tasks.emplace_back(node, level);
            }
        }

        if (node_level_tasks.empty())
        {
            printf(" Edge insertion ( )...\n");
            return 0;
        }

        // level
        std::vector<std::atomic<long long>> level_add_counts(static_cast<size_t>(level_limit));
        for (auto &counter : level_add_counts)
        {
            counter.store(0, std::memory_order_relaxed);
        }

        const size_t total_tasks = node_level_tasks.size();
        std::atomic<bool> half_progress_reported{false};
        std::atomic<size_t> processed_counter(0);

        int total_add_count = 0;
        int completed_tasks = 0;
        int failed_tasks = 0;

        printf(" : ->Edge insertion->remove ( %zu node-level )...\n", total_tasks);

#pragma omp parallel for schedule(dynamic, 32) num_threads(num_threads) reduction(+ : total_add_count, completed_tasks, failed_tasks)
        for (int task_idx = 0; task_idx < static_cast<int>(total_tasks); ++task_idx)
        {
            const auto &task = node_level_tasks[static_cast<size_t>(task_idx)];
            int start_id = task.first;
            int level = task.second;
            int node_level = get_node_level(index, start_id);
            if (level < 0 || level > node_level)
            {
                failed_tasks++;
                processed_counter.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            int local_add_count = process_single_tree_and_add_edges(
                base[start_id], start_id, efq_maintree, jump_max, repeat_max, level);

            if (local_add_count >= 0)
            {
                total_add_count += local_add_count;
                completed_tasks++;
                if (level >= 0 && level < static_cast<int>(level_add_counts.size()))
                {
                    level_add_counts[static_cast<size_t>(level)].fetch_add(local_add_count, std::memory_order_relaxed);
                }
            }
            else
            {
                failed_tasks++;
            }

            size_t current_processed = processed_counter.fetch_add(1, std::memory_order_relaxed) + 1;
            if (!half_progress_reported.load(std::memory_order_relaxed) &&
                current_processed * 2 >= total_tasks)
            {
                bool expected = false;
                if (half_progress_reported.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
                {
#pragma omp critical(progress_report)
                    {
                        float progress = static_cast<float>(current_processed) / static_cast<float>(total_tasks) * 100.0f;
                        printf(" : totaledge =%d, node =%d (%.2f%%)\n",
                               progress, current_processed, total_tasks);
                        fflush(stdout);
                    }
                }
            }
        }

        t.stop();
        printf(" Edge insertion ( ), total : %.2f \n", t.get());
        printf("total Edge insertion : %d, : %d, : %d\n", total_add_count, completed_tasks, failed_tasks);

        for (int level = std::min(level_limit - 1, max_global_level); level >= 0; --level)
        {
            long long level_added = level_add_counts[static_cast<size_t>(level)].load(std::memory_order_relaxed);
            if (level_added > 0)
            {
                printf(" Level %d addedge=%lld\n", level, level_added);
            }
        }

        printf(" : , %zu \n", base.num_);
        return total_add_count;
    }
};

// edge - remove edge
class EdgeImpactTester
{
private:
    IndexType &index;
    TrackedSearchExecutor executor;
    const DataSetWrapper<float> &base;
    const vector<vector<pair<int, int>>> &edge_stats;
    const PseudoGroundTruth &train_gt_;
    vector<vector<int>> train_gt_sorted_cache_;
    size_t recall_eval_k_;
    int weighted_discarded_threshold_ = 100;
    int max_base_points_per_edge_ = 100;

public:
    EdgeImpactTester(IndexType &idx, EdgeSearchTracker &tracker,
                     const DataSetWrapper<float> &base_data,
                     const vector<vector<pair<int, int>>> &stats,
                     const PseudoGroundTruth &train_gt,
                     size_t recall_eval_k,
                     int weighted_discarded_threshold = 100,
                     int max_base_points_per_edge = 100)
        : index(idx), executor(idx, tracker), base(base_data), edge_stats(stats), train_gt_(train_gt),
          train_gt_sorted_cache_(train_gt.num),
          recall_eval_k_(std::min(recall_eval_k, train_gt.dim)),
          weighted_discarded_threshold_(weighted_discarded_threshold),
          max_base_points_per_edge_(max_base_points_per_edge)
    {
        if (recall_eval_k_ == 0 && train_gt.dim > 0)
        {
            recall_eval_k_ = train_gt.dim;
        }
    }

    size_t get_recall_eval_k() const
    {
        return recall_eval_k_;
    }

    // edge
    struct EdgeImpactResult
    {
        int from, to;
        long long comparisons_with_edge;
        long long comparisons_without_edge;
        long long comparison_diff;
        double impact_ratio;       // |comparison_diff| / comparisons_with_edge
        bool should_delete;
        double avg_recall_with_edge;
        double avg_recall_without_edge;
        double recall_delta_pct;
        int sampled_base_points;
    };

    // edge - 24 ( baseline )

    // edge - 24 ( baseline )
    vector<EdgeImpactResult> test_multiple_edges(
        const vector<pair<int, int>> &candidate_edges,
        EdgeSearchTracker &tracker,
        int efq = 100, int k = 10)
    {
        if (candidate_edges.empty())
        {
            printf(" : Edge insertion remove, \n");
            return {};
        }

        size_t total_edges = candidate_edges.size();
        vector<EdgeImpactResult> results;
        results.reserve(total_edges);

        printf(" %zu candidateedge (24 , baseline )...\n", total_edges);

        // 
        printf(" levelEdge insertion: level Edge insertion. \n");
        auto [baseline_comps, baseline_recalls] = precompute_baseline_data(candidate_edges, tracker, efq);
        size_t sample_for_logging = std::min<size_t>(10, total_edges);
        printf(" %zu edgefor , edgefor . \n", sample_for_logging);

        // 
        printf(" levelEdge insertion: level Edge insertion. \n");
        atomic<int> completed_tests(0);
        int total_tests = static_cast<int>(candidate_edges.size());
        const int progress_step = 10;
        std::atomic<int> next_progress_percent(progress_step);

#pragma omp parallel num_threads(24)
        {
            // local
            vector<EdgeImpactResult> local_results;
            local_results.reserve(candidate_edges.size() / 24 + 1);

#pragma omp for schedule(dynamic, 4)
            for (size_t i = 0; i < candidate_edges.size(); i++)
            {
                int from = candidate_edges[i].first;
                int to = candidate_edges[i].second;

                // edge
                const vector<int> &base_points = tracker.get_base_points_for_edge(from, to);

                if (base_points.empty())
                {
                    // 
                    continue;
                }

                // baseline edge
                EdgeImpactResult result = test_edge_impact_cached(from, to, base_points, baseline_comps, baseline_recalls, efq, k);
                local_results.emplace_back(result);

                // : fixed
                int current_completed = completed_tests.fetch_add(1) + 1;
                if (total_tests > 0)
                {
                    double progress_ratio = static_cast<double>(current_completed) / static_cast<double>(total_tests);
                    int progress_percent = static_cast<int>(progress_ratio * 100.0);
                    int expected = next_progress_percent.load(std::memory_order_relaxed);
                    while (progress_percent >= expected)
                    {
                        int milestone = expected;
                        if (next_progress_percent.compare_exchange_strong(expected,
                                                                          milestone + progress_step, std::memory_order_acq_rel))
                        {
                            printf(" Progress reached ~50%%: %.1f%% (%zu/%zu)\n",
                                   milestone, current_completed, total_tests,
                                   progress_ratio * 100.0);
                            break;
                        }
                    }
                }
            }

// local global ( )
#pragma omp critical(merge_results)
            {
                results.insert(results.end(), local_results.begin(), local_results.end());
            }
        }

        // 
        int should_delete_count = 0;
        // add: impact_ratio edge
        int count_gt_1 = 0;
        int count_gt_1_5 = 0;
        int count_gt_2 = 0;
        int count_gt_5 = 0;
        for (const auto &result : results)
        {
            if (result.should_delete)
                should_delete_count++;
            if (result.impact_ratio > 0.01)
                count_gt_1++;
            if (result.impact_ratio > 0.015)
                count_gt_1_5++;
            if (result.impact_ratio > 0.02)
                count_gt_2++;
            if (result.impact_ratio > 0.05)
                count_gt_5++;
        }

        printf(" nodelevel , skip Edge insertion\n");
        printf(" edge : %zu\n", results.size());
        printf(" removeedge : %d\n", should_delete_count);
        printf(" Progress reached ~50%%: %.1f%% (%zu/%zu)\n",
               results.empty() ? 0.0 : (should_delete_count * 100.0 / results.size()));
        printf(" Edges with impact_ratio > 1%%: %d (%.2f%%)\n", count_gt_1, results.empty() ? 0.0 : (count_gt_1 * 100.0 / results.size()));
        printf(" impact_ratio > 1.5%% edge : %d (%.2f%%)\n", count_gt_1_5, results.empty() ? 0.0 : (count_gt_1_5 * 100.0 / results.size()));
        printf(" impact_ratio > 2%% edge : %d (%.2f%%)\n", count_gt_2, results.empty() ? 0.0 : (count_gt_2 * 100.0 / results.size()));
        printf(" impact_ratio > 5%% edge : %d (%.2f%%)\n", count_gt_5, results.empty() ? 0.0 : (count_gt_5 * 100.0 / results.size()));

        // add: impact_ratio /median/max
        if (!results.empty())
        {
            vector<double> ratios;
            ratios.reserve(results.size());
            double sum_ratio = 0.0;
            double max_ratio = 0.0;
            for (const auto &r : results)
            {
                double val = r.impact_ratio;
                ratios.push_back(val);
                sum_ratio += val;
                if (val > max_ratio)
                    max_ratio = val;
            }
            double mean_ratio = sum_ratio / ratios.size();
            sort(ratios.begin(), ratios.end());
            double median_ratio = 0.0;
            size_t n = ratios.size();
            if (n % 2 == 1)
                median_ratio = ratios[n / 2];
            else
                median_ratio = (ratios[n / 2 - 1] + ratios[n / 2]) / 2.0;
            printf(" Progress reached ~50%%: %.1f%% (%zu/%zu)\n",
                   mean_ratio * 100.0, median_ratio * 100.0, max_ratio * 100.0);
        }

        return results;
    }

private:
    void prepare_ground_truth_cache(const vector<int> &base_indices, size_t need_k)
    {
        size_t effective_k = std::min(need_k, static_cast<size_t>(train_gt_.dim));
        if (effective_k == 0)
        {
            return;
        }
        for (int idx : base_indices)
        {
            if (idx < 0 || idx >= static_cast<int>(train_gt_.num))
            {
                continue;
            }
            auto &cache = train_gt_sorted_cache_[idx];
            if (cache.size() >= effective_k)
            {
                continue;
            }
            const int *gt_row = train_gt_.row(idx);
            cache.assign(gt_row, gt_row + effective_k);
            std::sort(cache.begin(), cache.end());
        }
    }

    double compute_single_query_recall(int base_idx, const vector<int> &results, size_t k) const
    {
        if (k == 0)
        {
            return 0.0;
        }

        size_t eval_k = std::min(k, results.size());
        if (eval_k == 0)
        {
            return 0.0;
        }

        if (base_idx < 0 || base_idx >= static_cast<int>(train_gt_.num))
        {
            return 0.0;
        }

        const int *gt_row = train_gt_.row(base_idx);
        if (!gt_row)
        {
            return 0.0;
        }

        size_t gt_k = std::min(k, static_cast<size_t>(train_gt_.dim));
        size_t effective_gt = 0;
        for (size_t j = 0; j < gt_k; ++j)
        {
            if (gt_row[j] != base_idx)
            {
                ++effective_gt;
            }
        }
        if (effective_gt == 0)
        {
            return 0.0;
        }

        const vector<int> *cached_gt = nullptr;
        if (base_idx >= 0 && base_idx < static_cast<int>(train_gt_sorted_cache_.size()))
        {
            const auto &candidate_cache = train_gt_sorted_cache_[base_idx];
            if (!candidate_cache.empty())
            {
                cached_gt = &candidate_cache;
            }
        }

        size_t hits = 0;
        if (cached_gt)
        {
            for (size_t i = 0; i < eval_k; ++i)
            {
                int candidate = results[i];
                if (candidate == base_idx)
                {
                    continue;
                }
                if (std::binary_search(cached_gt->begin(), cached_gt->end(), candidate))
                {
                    ++hits;
                }
            }
            return (effective_gt > 0)
                       ? static_cast<double>(hits) / static_cast<double>(effective_gt)
                       : 0.0;
        }

        for (size_t i = 0; i < eval_k; ++i)
        {
            int candidate = results[i];
            if (candidate == base_idx)
            {
                continue;
            }
            for (size_t j = 0; j < gt_k; ++j)
            {
                int gt_val = gt_row[j];
                if (gt_val == base_idx)
                {
                    continue;
                }
                if (gt_val == candidate)
                {
                    ++hits;
                    break;
                }
            }
        }

        return static_cast<double>(hits) / static_cast<double>(effective_gt);
    }

    // : , “ edge”distance evaluations( NSG )
    pair<vector<long long>, vector<double>> precompute_baseline_data(
        const vector<pair<int, int>> &candidate_edges,
        EdgeSearchTracker &tracker, int efq)
    {
        // candidateedge ( edge )
        vector<char> needed(base.num_, 0);
        vector<int> unique_bases;
        unique_bases.reserve(1024);

        for (const auto &e : candidate_edges)
        {
            const auto &bp = tracker.get_base_points_for_edge(e.first, e.second);
            for (int b : bp)
                if (b >= 0 && b < base.num_)
                {
                    if (!needed[b])
                    {
                        needed[b] = 1;
                        unique_bases.emplace_back(b);
                    }
                }
        }

        // Ground Truth k , for recall
        prepare_ground_truth_cache(unique_bases, recall_eval_k_);

        // baselinedistance evaluations
        vector<long long> baseline(base.num_, -1);
        vector<double> baseline_recall(base.num_, -1.0);
#pragma omp parallel for schedule(dynamic, 1024) num_threads(24)
        for (size_t i = 0; i < unique_bases.size(); ++i)
        {
            int bidx = unique_bases[i];
            // ( edge , visited ), k=recall_eval_k_for recall
            thread_local vector<int> results;
            long long comp = executor.execute_search_baseline_fast(base[bidx], bidx, efq,
                                                                   static_cast<int>(recall_eval_k_), results);
            baseline[bidx] = comp;
            // baseline recall@recall_eval_k_
            baseline_recall[bidx] = compute_single_query_recall(bidx, results, recall_eval_k_);
            static thread_local int printed = 0;
            if (baseline_recall[bidx] <= 0.0 && printed < 5)
            {
                const int *gt_row = (bidx >= 0 && bidx < static_cast<int>(train_gt_.num)) ? train_gt_.row(bidx) : nullptr;
                printf(" impact_ratio stats: mean=%.4f%%, median=%.4f%%, max=%.4f%%\n",
                       bidx, baseline_recall[bidx], results.size(),
                       results.empty() ? -1 : results[0],
                       gt_row ? gt_row[0] : -1,
                       gt_row ? gt_row[1] : -1);
                printed++;
            }
        }

        printf(" baseline : =%zu( distance evaluations recall@%zu)\n", unique_bases.size(), recall_eval_k_);
        return {baseline, baseline_recall};
    }

    // baseline, edge
    EdgeImpactResult test_edge_impact_cached(
        int from, int to, const vector<int> &base_point_indices,
        const vector<long long> &baseline_comps, const vector<double> &baseline_recalls,
        int efq = 100, int k = 10)
    {
        EdgeImpactResult result{};
        result.from = from;
        result.to = to;
        result.comparisons_with_edge = 0;
        result.comparisons_without_edge = 0;

        struct RecallSample
        {
            int base_idx;
            double recall_with;
            double recall_without;
        };
        std::vector<RecallSample> recall_samples;
        recall_samples.reserve(3);

        // “ edge” ; “ edge” distance evaluations recall baseline
        thread_local vector<int> results_wo;
        double recall_with_sum = 0.0;
        double recall_without_sum = 0.0;
        int test_count = 0;
        for (int base_idx : base_point_indices)
        {
            if (base_idx < 0 || base_idx >= base.num_)
                continue;
            if (test_count >= max_base_points_per_edge_)
                break;
            // total“ edge”distance evaluations( )
            long long comp_with = baseline_comps[base_idx];
            double recall_with = (baseline_recalls.size() > (size_t)base_idx) ? baseline_recalls[base_idx] : -1.0;
            if (comp_with < 0 || recall_with < 0.0)
            {
                vector<int> tmp_res;
                comp_with = executor.execute_search_baseline_fast(base[base_idx], base_idx, efq,
                                                                  static_cast<int>(recall_eval_k_), tmp_res);
                result.comparisons_with_edge += comp_with;
                recall_with = compute_single_query_recall(base_idx, tmp_res, recall_eval_k_);
            }
            else
            {
                result.comparisons_with_edge += comp_with;
            }
            if (recall_with < 0.0)
            {
                recall_with = 0.0;
            }
            recall_with_sum += recall_with;
            // “ edge(from->to)” distance evaluations
            long long comp_without = executor.execute_search_disable_single_edge_fast(
                base[base_idx], base_idx, efq, static_cast<int>(recall_eval_k_), results_wo, from, to);
            result.comparisons_without_edge += comp_without;
            // edge recall@recall_eval_k_
            double recall_without = compute_single_query_recall(base_idx, results_wo, recall_eval_k_);
            recall_without_sum += recall_without;

            if (recall_samples.size() < 3)
            {
                recall_samples.push_back({base_idx, recall_with, recall_without});
            }
            test_count++;
        }

        // : edge distance evaluations (without < with) remove
        // edge
        int neighbor_idx = -1;
        const auto *neighbors_from = get_base_neighbors_const(index, from);
        if (neighbors_from)
        {
            for (int i = 0; i < static_cast<int>(neighbors_from->size()); ++i)
            {
                if ((*neighbors_from)[i] == to)
                {
                    neighbor_idx = i;
                    break;
                }
            }
        }
        int adopted = 0, discarded = 0;
        if (neighbor_idx >= 0 && neighbor_idx < edge_stats[from].size())
        {
            adopted = edge_stats[from][neighbor_idx].first;
            discarded = edge_stats[from][neighbor_idx].second;
        }
        int total_accesses = adopted + discarded;
        // 
        double weighted_discarded = 0.0;
        if (total_accesses > 0)
        {
            double sample_ratio = (double)test_count / (double)total_accesses;
            // double used_discarded = discarded * sample_ratio;
            if (total_accesses > weighted_discarded_threshold_)
                weighted_discarded = (double(weighted_discarded_threshold_) / total_accesses) * discarded;
            else
                weighted_discarded = discarded;
        }

        result.comparison_diff = result.comparisons_without_edge - result.comparisons_with_edge - weighted_discarded;

        // (recall@K): avgrecall ±1% distance evaluations
        double avg_recall_with = (test_count > 0) ? (recall_with_sum / test_count) : 0.0;
        double avg_recall_without = (test_count > 0) ? (recall_without_sum / test_count) : 0.0;
        double recall_delta_pct = (avg_recall_without - avg_recall_with) * 100.0;
        // : / 0.2%, / 1%distance evaluations( )
        double quality_adjustment = 0.0;
        if (std::abs(recall_delta_pct) >= 0.01)
        {
            double ratio = recall_delta_pct / 0.2;
            quality_adjustment = -0.01 * ratio * (double)result.comparisons_with_edge;
            long long qa = llround(quality_adjustment);
            result.comparison_diff += qa;
        }
        result.avg_recall_with_edge = avg_recall_with;
        result.avg_recall_without_edge = avg_recall_without;
        result.recall_delta_pct = recall_delta_pct;
        result.sampled_base_points = test_count;
        double raw_diff = static_cast<double>(result.comparisons_without_edge) - static_cast<double>(result.comparisons_with_edge);
        result.impact_ratio = (result.comparisons_with_edge > 0)
                                  ? std::fabs(raw_diff) / static_cast<double>(result.comparisons_with_edge)
                                  : 0.0;

        static std::atomic<int> global_recall_samples_printed{0};
        constexpr int kMaxGlobalRecallSamples = 6;
        if (!recall_samples.empty())
        {
            for (const auto &sample : recall_samples)
            {
                int expected = global_recall_samples_printed.load(std::memory_order_relaxed);
                while (expected < kMaxGlobalRecallSamples)
                {
                    if (global_recall_samples_printed.compare_exchange_strong(
                            expected, expected + 1, std::memory_order_acq_rel))
                    {
#pragma omp critical(edge_recall_sample_print)
                        {
                            printf(" impact_ratio stats: mean=%.4f%%, median=%.4f%%, max=%.4f%%\n",
                                   from, to, sample.base_idx, recall_eval_k_,
                                   sample.recall_with, sample.recall_without);
                        }
                        break;
                    }
                }
            }
        }
        // should_delete ,
        result.should_delete = false;
        return result;
    }
};

class NewEdgePruningAlgorithm
{
private:
    IndexType &index;
    const DataSetWrapper<float> &base;
    OptimizedEdgePruner &original_pruner;
    const PseudoGroundTruth &train_gt_;
    double sampling_ratio_ = 1.0;

public:
    NewEdgePruningAlgorithm(IndexType &idx,
                            const DataSetWrapper<float> &base_data,
                            OptimizedEdgePruner &pruner,
                            const PseudoGroundTruth &train_gt)
        : index(idx), base(base_data), original_pruner(pruner), train_gt_(train_gt) {}

    void set_sampling_ratio(double ratio)
    {
        double sanitized = ratio;
        if (ratio <= 0.0)
        {
            printf("Warning: %.4f , . \n", ratio);
            sanitized = 1.0;
        }
        sanitized = min(1.0, sanitized);
        double previous = sampling_ratio_;
        sampling_ratio_ = sanitized;
        if (std::fabs(previous - sampling_ratio_) < 1e-12)
        {
            return;
        }
        if (sampling_ratio_ >= 0.9999)
        {
            printf(" Pruning : seed set. \n");
        }
        else
        {
            printf(" [Recall sample] edge %d->%d, base %d, Recall@%zu: with edge=%.4f, without edge=%.4f\n",
                   sampling_ratio_, sampling_ratio_ * 100.0);
        }
    }

    // edge
    int execute_new_pruning_algorithm(float prune_ratio, int efq_for_tracking = 100,
                                      int efq_for_testing = 100, int k = 10, int total_batches = 1,
                                      int allowed_levels = 1)
    {
        const int level_limit = std::max(1, allowed_levels);
        printf(" edge ...( level: 0~%d)\n", level_limit - 1);
        vector<int> sampled_base_points = select_base_points();
        if (sampled_base_points.empty())
        {
            sampled_base_points.resize(base.num_);
            iota(sampled_base_points.begin(), sampled_base_points.end(), 0);
            printf(" Pruning : seed set. \n");
        }
        const size_t sampled_count = sampled_base_points.size();
        const double sampled_pct = base.num_ > 0 ? (100.0 * sampled_count / base.num_) : 0.0;
        const bool use_full_dataset = sampled_pct >= 99.9;

        printf(" [Recall sample] edge %d->%d, base %d, Recall@%zu: with edge=%.4f, without edge=%.4f\n",
               prune_ratio, efq_for_tracking, efq_for_testing, k, total_batches,
               sampled_count, base.num_, sampled_pct);

        Timer total_timer;
        total_timer.start();

        // 
        printf(" [Recall sample] edge %d->%d, base %d, Recall@%zu: with edge=%.4f, without edge=%.4f\n",
               use_full_dataset ? " " : " ");
        // 
        MultiLevelEdgeStats edge_stats_for_all =
            original_pruner.build_edge_statistics_optimized(base, efq_for_tracking, total_batches, &sampled_base_points, level_limit);

        const int max_level_global = std::max(0, std::min(get_max_level(index), level_limit - 1));
        std::vector<EdgeKey> adopted_zero_edges;
        adopted_zero_edges.reserve(1024);
        std::vector<long long> per_level_total_edges(static_cast<size_t>(max_level_global) + 1, 0);
        std::vector<long long> per_level_adopted_zero(static_cast<size_t>(max_level_global) + 1, 0);
        std::vector<long long> per_level_adopted_zero_with_discard(static_cast<size_t>(max_level_global) + 1, 0);

        int max_nodes_for_stats = std::min(static_cast<int>(edge_stats_for_all.size()), static_cast<int>(index.link_lists_.size()));
        for (int node = 0; node < max_nodes_for_stats; ++node)
        {
            const auto &node_levels_stats = edge_stats_for_all[node];
            const auto &node_levels_links = index.link_lists_[node];
            size_t level_count = std::min(node_levels_stats.size(), node_levels_links.size());
            level_count = std::min(level_count, static_cast<size_t>(level_limit));
            for (size_t level = 0; level < level_count; ++level)
            {
                const auto *neighbors = get_neighbors_const(index, node, static_cast<int>(level));
                const size_t neighbor_count = neighbors ? std::min(node_levels_stats[level].size(), neighbors->size()) : 0;
                if (level < per_level_total_edges.size())
                {
                    per_level_total_edges[level] += static_cast<long long>(neighbor_count);
                }
                for (size_t ni = 0; ni < neighbor_count; ++ni)
                {
                    const auto &stat = node_levels_stats[level][ni];
                    const int adopted = stat.first;
                    const int discarded = stat.second;
                    const int neighbor_id = (*neighbors)[ni];
                    if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                    {
                        continue;
                    }
                    if (adopted == 0 && level < per_level_adopted_zero.size())
                    {
                        per_level_adopted_zero[level]++;
                        if (discarded > 0)
                        {
                            per_level_adopted_zero_with_discard[level]++;
                            adopted_zero_edges.emplace_back(node, neighbor_id, static_cast<int>(level));
                        }
                    }
                }
            }
        }

        for (int level = 0; level <= max_level_global; ++level)
        {
            printf(" [Recall sample] edge %d->%d, base %d, Recall@%zu: with edge=%.4f, without edge=%.4f\n",
                   level,
                   per_level_total_edges[static_cast<size_t>(level)],
                   per_level_adopted_zero[static_cast<size_t>(level)],
                   per_level_adopted_zero_with_discard[static_cast<size_t>(level)]);
        }

        // candidateedge( )
        vector<EdgeKey> candidate_edges = select_candidate_edges_from_stats(edge_stats_for_all, prune_ratio, level_limit);
        printf(" candidateedge: %zu ( level)\n", candidate_edges.size());

        printf(" Pruning : seed set. \n");
        std::vector<EdgeKey> edges_to_delete = candidate_edges;
        edges_to_delete.insert(edges_to_delete.end(), adopted_zero_edges.begin(), adopted_zero_edges.end());
        if (edges_to_delete.empty())
        {
            printf(" Pruning : seed set. \n");
            total_timer.stop();
            return 0;
        }

        std::vector<long long> per_level_total_candidates(static_cast<size_t>(max_level_global) + 1, 0);
        for (const auto &edge : edges_to_delete)
        {
            if (edge.level >= 0 && edge.level < static_cast<int>(per_level_total_candidates.size()))
            {
                per_level_total_candidates[static_cast<size_t>(edge.level)]++;
            }
        }

        // dedup remove( level)
        std::unordered_set<EdgeKey, EdgeKeyHash> unique_edges;
        unique_edges.reserve(edges_to_delete.size() * 2 + 1);
        std::vector<EdgeKey> deduped_edges;
        deduped_edges.reserve(edges_to_delete.size());
        std::vector<long long> deduped_per_level(static_cast<size_t>(max_level_global) + 1, 0);
        for (const auto &edge : edges_to_delete)
        {
            if (edge.from < 0 || edge.from >= index.base_.num_ ||
                edge.to < 0 || edge.to >= index.base_.num_)
            {
                continue;
            }
            if (edge.level < 0 || edge.level >= static_cast<int>(index.link_lists_[edge.from].size()))
            {
                continue;
            }
            if (unique_edges.insert(edge).second)
            {
                deduped_edges.emplace_back(edge);
                if (edge.level >= 0 && edge.level < static_cast<int>(deduped_per_level.size()))
                {
                    deduped_per_level[static_cast<size_t>(edge.level)]++;
                }
            }
        }

        const int max_delete_per_node_per_level = 2;
        std::vector<std::vector<int>> node_delete_count(index.link_lists_.size());
        for (size_t node = 0; node < index.link_lists_.size(); ++node)
        {
            size_t level_count = std::min(index.link_lists_[node].size(), static_cast<size_t>(level_limit));
            node_delete_count[node].assign(level_count, 0);
        }

        std::vector<long long> deleted_per_level(static_cast<size_t>(max_level_global) + 1, 0);
        std::vector<long long> blocked_per_level(static_cast<size_t>(max_level_global) + 1, 0);
        int deleted_count = 0;
        int blocked_count = 0;
        for (const auto &edge : deduped_edges)
        {
            const int node = edge.from;
            const int neighbor = edge.to;
            const int level = edge.level;
            if (node < 0 || node >= static_cast<int>(node_delete_count.size()))
            {
                continue;
            }
            if (level < 0 || level >= static_cast<int>(node_delete_count[node].size()))
            {
                continue;
            }
            if (node_delete_count[node][level] >= max_delete_per_node_per_level)
            {
                blocked_count++;
                if (level >= 0 && level < static_cast<int>(blocked_per_level.size()))
                {
                    blocked_per_level[static_cast<size_t>(level)]++;
                }
                continue;
            }

            auto *neighbors = get_neighbors_mut(index, node, level);
            if (!neighbors)
            {
                continue;
            }

            auto it = std::find(neighbors->begin(), neighbors->end(), neighbor);
            if (it == neighbors->end())
            {
                continue;
            }

            neighbors->erase(it);
            node_delete_count[node][level]++;
            deleted_count++;
            if (level >= 0 && level < static_cast<int>(deleted_per_level.size()))
            {
                deleted_per_level[static_cast<size_t>(level)]++;
            }
        }

        total_timer.stop();
        printf(" Level %d: totaledge=%lld, adopted==0=%lld ( discarded>0=%lld)\n",
               deleted_count, blocked_count, total_timer.get());

        for (int level = 0; level <= max_level_global; ++level)
        {
            printf(" Level %d: totaledge=%lld, adopted==0=%lld ( discarded>0=%lld)\n",
                   level,
                   per_level_total_candidates[static_cast<size_t>(level)],
                   deduped_per_level[static_cast<size_t>(level)],
                   deleted_per_level[static_cast<size_t>(level)],
                   blocked_per_level[static_cast<size_t>(level)]);
        }

        return deleted_count;
    }

private:
    // for (support )
    vector<int> select_base_points()
    {
        vector<int> base_points;
        base_points.resize(base.num_);
        iota(base_points.begin(), base_points.end(), 0);

        if (sampling_ratio_ >= 0.9999 || base_points.empty())
        {
            printf(" %zu base \n", base_points.size());
            return base_points;
        }

        size_t sample_size = static_cast<size_t>(std::round(static_cast<double>(base_points.size()) * sampling_ratio_));
        sample_size = max<size_t>(1, min(sample_size, base_points.size()));

        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(base_points.begin(), base_points.end(), rng);
        base_points.resize(sample_size);
        std::sort(base_points.begin(), base_points.end());

        printf(" %zu base ( %.2f%%)for \n", base_points.size(),
               base_points.empty() ? 0.0 : (100.0 * base_points.size() / base.num_));
        return base_points;
    }

    // node candidateedge( PIGR9_6clear_2 Pruning )
    vector<EdgeKey> select_candidate_edges_by_existing_metrics(float prune_ratio, int efq_maintree, int total_batches)
    {
        (void)prune_ratio;
        printf(" candidateedge remove, . \n");

        // 1) edge
        MultiLevelEdgeStats edge_stats =
            original_pruner.build_edge_statistics_optimized(base, efq_maintree, total_batches);

        vector<EdgeKey> edges_with_access;
        edges_with_access.reserve(index.base_.num_);

        int total_valid_edges = 0;
        int edges_with_stats = 0;

        int max_nodes = std::min(static_cast<int>(edge_stats.size()), static_cast<int>(index.link_lists_.size()));
        for (int node = 0; node < max_nodes; node++)
        {
            const auto &node_levels_stats = edge_stats[static_cast<size_t>(node)];
            const auto &node_levels_links = index.link_lists_[node];
            const size_t level_count = std::min(node_levels_stats.size(), node_levels_links.size());
            for (size_t level = 0; level < level_count; ++level)
            {
                const auto *neighbors = get_neighbors_const(index, node, static_cast<int>(level));
                const size_t neighbor_count = neighbors ? std::min(node_levels_stats[level].size(), neighbors->size()) : 0;
                total_valid_edges += static_cast<int>(neighbor_count);
                for (size_t neighbor_idx = 0; neighbor_idx < neighbor_count; neighbor_idx++)
                {
                    const auto &stat = node_levels_stats[level][neighbor_idx];
                    int adopted = stat.first;
                    if (adopted <= 0)
                        continue;

                    int neighbor_id = (*neighbors)[neighbor_idx];
                    if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                        continue;

                    edges_with_access.emplace_back(node, neighbor_id, static_cast<int>(level));
                    edges_with_stats++;
                }
            }
        }

        if (edges_with_access.empty())
        {
            printf(" candidateedge remove, . \n");
            return {};
        }

        const float sample_ratio = 0.05f;
        size_t sample_size = static_cast<size_t>(edges_with_access.size() * sample_ratio);
        if (sample_size == 0)
            sample_size = 1;
        if (sample_size > edges_with_access.size())
            sample_size = edges_with_access.size();

        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(edges_with_access.begin(), edges_with_access.end(), rng);
        edges_with_access.resize(sample_size);

        printf(" (adopted>0) edge 5%% candidateedge...\n");
        printf(" totaledge : %d\n", total_valid_edges);
        printf(" Edges with adopted>0: %d\n", edges_with_stats);
        printf(" Sampling ratio: %.1f%%, actual sample count: %zu (cross-level)\n", sample_ratio * 100.0f, edges_with_access.size());

        return edges_with_access;
    }

    // candidateedge,
    vector<EdgeKey> select_candidate_edges_from_stats(
        const MultiLevelEdgeStats &edge_stats,
        float prune_ratio,
        int level_limit)
    {
        printf(" (adopted>0) edge 5%% candidateedge...\n");

        struct EdgeCandidate
        {
            int node;
            int neighbor;
            int level;
            int adopted;
            int discarded;
            int total;
            double ratio;
        };

        const int effective_level_limit = (level_limit <= 0) ? 1 : level_limit;
        const int max_level = std::max(0, std::min(get_max_level(index), effective_level_limit - 1));
        std::vector<std::vector<EdgeCandidate>> candidates_per_level(static_cast<size_t>(max_level) + 1);

        std::vector<long long> per_level_total_valid(static_cast<size_t>(max_level) + 1, 0);
        std::vector<long long> per_level_with_stats(static_cast<size_t>(max_level) + 1, 0);
        std::vector<long long> per_level_adopted_zero(static_cast<size_t>(max_level) + 1, 0);
        std::vector<long long> per_level_adopted_zero_with_discard(static_cast<size_t>(max_level) + 1, 0);

        const size_t max_nodes = std::min(edge_stats.size(), index.link_lists_.size());
        for (size_t node = 0; node < max_nodes; ++node)
        {
            const auto &node_levels = edge_stats[node];
            const auto &link_levels = index.link_lists_[node];
            size_t level_count = std::min(node_levels.size(), link_levels.size());
            level_count = std::min(level_count, static_cast<size_t>(effective_level_limit));
            for (size_t level = 0; level < level_count; ++level)
            {
                const auto *neighbors = get_neighbors_const(index, static_cast<int>(node), static_cast<int>(level));
                const size_t neighbor_count = neighbors ? std::min(node_levels[level].size(), neighbors->size()) : 0;
                if (level >= per_level_total_valid.size())
                {
                    continue;
                }
                per_level_total_valid[level] += static_cast<long long>(neighbor_count);
                for (size_t idx = 0; idx < neighbor_count; ++idx)
                {
                    const auto &stat = node_levels[level][idx];
                    const int neighbor_id = (*neighbors)[idx];
                    if (neighbor_id < 0 || neighbor_id >= index.base_.num_)
                    {
                        continue;
                    }
                    const int adopted = stat.first;
                    const int discarded = stat.second;
                    if (adopted > 0)
                    {
                        per_level_with_stats[level]++;
                        const double ratio = static_cast<double>(discarded) / static_cast<double>(adopted);
                        candidates_per_level[level].push_back({static_cast<int>(node), neighbor_id, static_cast<int>(level), adopted, discarded, adopted + discarded, ratio});
                    }
                    else
                    {
                        per_level_adopted_zero[level]++;
                        if (discarded > 0)
                        {
                            per_level_adopted_zero_with_discard[level]++;
                        }
                    }
                }
            }
        }

        vector<EdgeKey> selected;
        double ratio_clamped = std::max(0.0f, std::min(prune_ratio, 1.0f));
        for (int level = 0; level <= max_level; ++level)
        {
            auto &candidates = candidates_per_level[static_cast<size_t>(level)];
            if (candidates.empty())
            {
                continue;
            }

            std::sort(candidates.begin(), candidates.end(), [](const EdgeCandidate &a, const EdgeCandidate &b)
                      {
                if (std::fabs(a.ratio - b.ratio) > 1e-12) return a.ratio > b.ratio;
                if (a.total != b.total) return a.total > b.total;
                return a.discarded > b.discarded; });

            size_t total_candidates = candidates.size();
            size_t target = static_cast<size_t>(std::ceil(total_candidates * ratio_clamped));
            if (target == 0)
            {
                target = 1;
            }

            unordered_set<int> used_nodes;
            size_t selected_this_level = 0;
            for (const auto &entry : candidates)
            {
                if (used_nodes.insert(entry.node).second)
                {
                    selected.emplace_back(entry.node, entry.neighbor, entry.level);
                    selected_this_level++;
                    if (selected_this_level >= target)
                    {
                        break;
                    }
                }
            }

            printf(" Level %d: candidate=%lld, dedup =%lld, remove=%lld, =%lld\n",
                   level, total_candidates, selected_this_level, ratio_clamped * 100.0,
                   per_level_with_stats[static_cast<size_t>(level)],
                   per_level_adopted_zero[static_cast<size_t>(level)],
                   per_level_adopted_zero_with_discard[static_cast<size_t>(level)]);
        }

        return selected;
    }

    void print_edge_comparison_examples(const vector<EdgeImpactTester::EdgeImpactResult> &results,
                                        size_t recall_k,
                                        size_t max_examples = 10)
    {
        size_t detail_count = std::min(max_examples, results.size());
        if (detail_count == 0)
        {
            printf(" adopted/adopted candidateedge( , level )...\n");
            return;
        }

        vector<size_t> sample_indices(results.size());
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::mt19937 rng((unsigned int)std::random_device{}());
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

        printf("\n=== %zu edge ===\n", detail_count);
        for (size_t i = 0; i < detail_count; ++i)
        {
            const auto &r = results[sample_indices[i]];
            printf("#%zu edge %d->%d ( =%d)\n", i + 1, r.from, r.to, r.sampled_base_points);
            printf(" Level %d: candidateedge=%zu, =%zu, target =%.2f%%, (adopted>0)=%lld, adopted==0=%lld( discarded>0=%lld)\n",
                   r.comparisons_with_edge, recall_k, r.avg_recall_with_edge);
            printf(" Level %d: candidateedge=%zu, =%zu, target =%.2f%%, (adopted>0)=%lld, adopted==0=%lld( discarded>0=%lld)\n",
                   r.comparisons_without_edge, recall_k, r.avg_recall_without_edge);
            printf(" Level %d: candidateedge=%zu, =%zu, target =%.2f%%, (adopted>0)=%lld, adopted==0=%lld( discarded>0=%lld)\n",
                   r.comparisons_without_edge - r.comparisons_with_edge,
                   r.recall_delta_pct,
                   r.impact_ratio * 100.0);
        }
    }

    // 
    void print_algorithm_statistics(const vector<EdgeImpactTester::EdgeImpactResult> &results)
    {
        if (results.empty())
        {
            printf(" adopted/adopted candidateedge( , level )...\n");
            return;
        }

        printf(" adopted/adopted candidateedge( , level )...\n");

        // 
        int total_tested = results.size();
        int should_delete = 0;
        int negative_impact = 0;
        int positive_impact = 0;

        long long total_comp_with = 0;
        long long total_comp_without = 0;
        vector<double> impact_ratios;
        vector<double> recall_delta_values;

        for (const auto &result : results)
        {
            total_comp_with += result.comparisons_with_edge;
            total_comp_without += result.comparisons_without_edge;
            impact_ratios.emplace_back(result.impact_ratio);
            recall_delta_values.emplace_back(result.avg_recall_without_edge - result.avg_recall_with_edge);

            if (result.should_delete)
                should_delete++;
            if (result.comparison_diff < 0)
                negative_impact++;
            if (result.comparison_diff > 0)
                positive_impact++;
        }

        // avg
        double avg_impact_ratio = accumulate(impact_ratios.begin(), impact_ratios.end(), 0.0) / impact_ratios.size();

        // add : 0 edge
        int negative_diff_count = 0;
        long long negative_diff_sum = 0LL;

        for (const auto &result : results)
        {
            if (result.comparison_diff < 0)
            {
                negative_diff_count++;
                negative_diff_sum += result.comparison_diff;
            }
        }

        printf(" edge : %d\n", total_tested);
        printf(" removeedge : %d (%.2f%%)\n", should_delete, should_delete * 100.0 / total_tested);
        printf(" edge : %d (%.2f%%)\n", negative_impact, negative_impact * 100.0 / total_tested);
        printf(" edge : %d (%.2f%%)\n", positive_impact, positive_impact * 100.0 / total_tested);
        printf(" <0 edge : %d (%.2f%%)\n", negative_diff_count, negative_diff_count * 100.0 / total_tested);
        printf(" <0 : %+lld\n", negative_diff_sum);
        printf(" edgedistance evaluations : %lld\n", total_comp_with);
        printf(" edgedistance evaluations : %lld\n", total_comp_with);
        printf("totaldistance evaluations( edge): %lld\n", total_comp_without);
        printf("avg : %.4f\n", avg_impact_ratio);
        printf(" Level %d: candidateedge=%zu, =%zu, target =%.2f%%, (adopted>0)=%lld, adopted==0=%lld( discarded>0=%lld)\n",
               total_comp_without - total_comp_with,
               total_comp_with > 0 ? (total_comp_without - total_comp_with) * 100.0 / total_comp_with : 0.0);

        if (!recall_delta_values.empty())
        {
            vector<double> sorted_deltas = recall_delta_values;
            sort(sorted_deltas.begin(), sorted_deltas.end());
            auto quantile = [&](double q)
            {
                if (sorted_deltas.empty())
                    return 0.0;
                if (sorted_deltas.size() == 1)
                    return sorted_deltas.front();
                double pos = q * (sorted_deltas.size() - 1);
                size_t lower = static_cast<size_t>(pos);
                size_t upper = std::min(sorted_deltas.size() - 1, lower + 1);
                double weight = pos - lower;
                return sorted_deltas[lower] * (1.0 - weight) + sorted_deltas[upper] * weight;
            };

            double min_delta = sorted_deltas.front();
            double max_delta = sorted_deltas.back();
            double mean_delta = accumulate(recall_delta_values.begin(), recall_delta_values.end(), 0.0) / recall_delta_values.size();
            double median_delta = quantile(0.5);
            double p25_delta = quantile(0.25);
            double p75_delta = quantile(0.75);

            printf(" With this edge: distance evaluations=%lld, avg Recall@%zu=%.4f\n",
                   min_delta, max_delta, mean_delta, median_delta, p25_delta, p75_delta);
            printf(" With this edge: distance evaluations=%lld, avg Recall@%zu=%.4f\n",
                   min_delta * 100.0, max_delta * 100.0, mean_delta * 100.0,
                   median_delta * 100.0, p25_delta * 100.0, p75_delta * 100.0);
        }
    }
};

// ======================== CSV ========================

class CSVDataReader
{
public:
    // CSV
    static vector<vector<string>> read_csv(const string &csv_path)
    {
        vector<vector<string>> data;
        ifstream file(csv_path);

        if (!file.is_open())
        {
            printf("Failed to open CSV file: %s\n", csv_path.c_str());
            return data;
        }

        string line;
        while (getline(file, line))
        {
            vector<string> row;
            stringstream ss(line);
            string cell;

            while (getline(ss, cell, ','))
            {
                row.emplace_back(cell);
            }

            if (!row.empty())
            {
                data.emplace_back(move(row));
            }
        }

        file.close();
        return data;
    }

    // CSV QPS
    static vector<float> extract_qps_values(const vector<string> &row, int start_col = 6, bool high_recall_only = false)
    {
        vector<float> qps_values;

        // recall (>=0.9), 10 (recall=0.9)
        int actual_start_col = high_recall_only ? 9 : start_col;

        for (int i = actual_start_col; i < row.size(); i++)
        {
            try
            {
                float qps = stof(row[i]);
                if (qps > 0)
                {
                    qps_values.emplace_back(qps);
                }
            }
            catch (const exception &)
            {
                // 
            }
        }

        return qps_values;
    }

    // CSV recall QPS ( recall >= 0.9)
    static vector<pair<float, float>> extract_recall_qps_pairs(const vector<string> &row)
    {
        vector<pair<float, float>> recall_qps_pairs;

        // recall 0.9 0.99, 9 15
        vector<float> recall_values = {0.9f, 0.92f, 0.94f, 0.96f, 0.97f, 0.98f, 0.99f};
        vector<int> col_indices = {9, 10, 11, 12, 13, 14, 15};

        for (size_t i = 0; i < recall_values.size() && i < col_indices.size(); i++)
        {
            int col_idx = col_indices[i];
            if (col_idx < row.size())
            {
                try
                {
                    float qps = stof(row[col_idx]);
                    if (qps > 0)
                    {
                        recall_qps_pairs.emplace_back(recall_values[i], qps);
                    }
                }
                catch (const exception &)
                {
                    // 
                }
            }
        }

        return recall_qps_pairs;
    }
};

// ======================== ========================

class AdaptiveStrategyManager
{
private:
    PerformanceTester &tester;
    string csv_path;

    // CSV QPS
    vector<float> read_recent_qps(int R, int efc, int k)
    {
        auto csv_data = CSVDataReader::read_csv(csv_path);
        vector<float> recent_qps;

        // 
        int max_iter = -1;
        vector<string> latest_row;

        // skip ,
        for (int i = 1; i < csv_data.size(); i++)
        {
            const auto &row = csv_data[i];
            if (row.size() < 4)
                continue;

            try
            {
                int csv_R = stoi(row[0]);
                int csv_efc = stoi(row[1]);
                int csv_iter = stoi(row[2]);
                int csv_k = stoi(row[3]);

                if (csv_R == R && csv_efc == efc && csv_k == k && csv_iter > max_iter)
                {
                    max_iter = csv_iter;
                    latest_row = row;
                }
            }
            catch (const exception &)
            {
                continue;
            }
        }

        // QPS ( recall >= 0.9)
        if (!latest_row.empty())
        {
            recent_qps = CSVDataReader::extract_qps_values(latest_row, 6, true);
            printf(" k=%d %d QPS : %zu ( recall >= 0.9)\n", k, max_iter, recent_qps.size());
        }

        return recent_qps;
    }

    // CSV recall-QPS
    vector<pair<float, float>> read_recent_recall_qps(int R, int efc, int k)
    {
        auto csv_data = CSVDataReader::read_csv(csv_path);
        vector<pair<float, float>> recent_recall_qps;

        // 
        int max_iter = -1;
        vector<string> latest_row;

        // skip ,
        for (int i = 1; i < csv_data.size(); i++)
        {
            const auto &row = csv_data[i];
            if (row.size() < 4)
                continue;

            try
            {
                int csv_R = stoi(row[0]);
                int csv_efc = stoi(row[1]);
                int csv_iter = stoi(row[2]);
                int csv_k = stoi(row[3]);

                if (csv_R == R && csv_efc == efc && csv_k == k && csv_iter > max_iter)
                {
                    max_iter = csv_iter;
                    latest_row = row;
                }
            }
            catch (const exception &)
            {
                continue;
            }
        }

        // QPS ( recall >= 0.9)
        if (!latest_row.empty())
        {
            recent_recall_qps = CSVDataReader::extract_recall_qps_pairs(latest_row);
            printf(" k=%d %d recall-QPS : %zu ( recall >= 0.9)\n",
                   k, max_iter, recent_recall_qps.size());
        }

        return recent_recall_qps;
    }

    // 
    vector<string> read_previous_iteration_record(int R, int efc, int k)
    {
        auto csv_data = CSVDataReader::read_csv(csv_path);
        vector<string> prev_row;
        int max_iter = -1;
        int second_max_iter = -1;

        // skip ,
        for (int i = 1; i < csv_data.size(); i++)
        {
            const auto &row = csv_data[i];
            if (row.size() < 4)
                continue;

            try
            {
                int csv_R = stoi(row[0]);
                int csv_efc = stoi(row[1]);
                int csv_iter = stoi(row[2]);
                int csv_k = stoi(row[3]);

                if (csv_R == R && csv_efc == efc && csv_k == k)
                {
                    if (csv_iter > max_iter)
                    {
                        second_max_iter = max_iter;
                        max_iter = csv_iter;
                    }
                    else if (csv_iter > second_max_iter && csv_iter < max_iter)
                    {
                        second_max_iter = csv_iter;
                    }
                }
            }
            catch (const exception &)
            {
                continue;
            }
        }

        // 
        if (second_max_iter >= 0)
        {
            for (int i = 1; i < csv_data.size(); i++)
            {
                const auto &row = csv_data[i];
                if (row.size() < 4)
                    continue;

                try
                {
                    int csv_R = stoi(row[0]);
                    int csv_efc = stoi(row[1]);
                    int csv_iter = stoi(row[2]);
                    int csv_k = stoi(row[3]);

                    if (csv_R == R && csv_efc == efc && csv_k == k && csv_iter == second_max_iter)
                    {
                        prev_row = row;
                        break;
                    }
                }
                catch (const exception &)
                {
                    continue;
                }
            }
            printf(" k=%d (iter=%d), (iter=%d)\n", k, second_max_iter, max_iter);
        }
        else
        {
            printf(" k=%d , ( %d )\n", k, max_iter >= 0 ? 1 : 0);
        }

        return prev_row;
    }

public:
    AdaptiveStrategyManager(PerformanceTester &test, const string &csv)
        : tester(test), csv_path(csv) {}

    // CSV
    void update_csv_path(const string &new_csv_path)
    {
        csv_path = new_csv_path;
        printf("Update CSV path to: %s\n", csv_path.c_str());
    }

    // efq : recall>=target_recall efq , recall/qps
    int find_optimal_efq_for_target_recall(IndexType &index,
                                           const DataSetWrapper<float> &dataset,
                                           const GroundTruth &gt,
                                           int k,
                                           double target_recall,
                                           int efq_start = 5,
                                           int coarse_step = 20,
                                           int refine_step = 2,
                                           int efq_max = 2000,
                                           float *achieved_recall = nullptr,
                                           float *achieved_qps = nullptr)
    {
        const float target = static_cast<float>(target_recall);
        Timer timer;
        index.set_num_threads(24);

        printf(" efq (k=%d, targetrecall=%.4f)...\n", k, target);

        int optimal_efq = -1;
        float optimal_recall = 0.0f;
        float optimal_qps = 0.0f;
        float last_recall = 0.0f;
        float last_qps = 0.0f;

        int lower_efq = -1;
        float lower_recall = 0.0f;
        float lower_qps = 0.0f;
        int upper_efq = -1;
        float upper_recall = 0.0f;
        float upper_qps = 0.0f;

        const int actual_start = std::max(efq_start, k);
        for (int efq = actual_start; efq <= efq_max; efq += coarse_step)
        {
            index.get_comparison_and_clear();
            timer.start();
            const int effective_efq = std::max(efq, k);
            auto [_, knn] = index.search(dataset, k, effective_efq);
            timer.stop();

            float recall = gt.recall(k, knn);
            float qps_val = dataset.num_ / timer.get();
            timer.reset();

            last_recall = recall;
            last_qps = qps_val;

            if (recall >= target)
            {
                upper_efq = efq;
                upper_recall = recall;
                upper_qps = qps_val;
                if (lower_efq == -1)
                {
                    lower_efq = efq - coarse_step;
                    if (lower_efq < actual_start)
                        lower_efq = actual_start;
                }
                break;
            }
            else
            {
                lower_efq = efq;
                lower_recall = recall;
                lower_qps = qps_val;
            }
        }

        if (upper_efq == -1)
        {
            optimal_efq = efq_max;
            optimal_recall = last_recall;
            optimal_qps = last_qps;
            printf(" range target efq, max%d (recall=%.4f, qps=%.1f)\n",
                   optimal_efq, last_recall, last_qps);
        }
        else
        {
            if (lower_efq < actual_start)
            {
                lower_efq = actual_start;
            }
            int refined_upper = upper_efq;
            float refined_recall = upper_recall;
            float refined_qps = upper_qps;

            for (int efq = lower_efq + refine_step; efq < upper_efq; efq += refine_step)
            {
                index.get_comparison_and_clear();
                timer.start();
                const int effective_efq = std::max(efq, k);
                auto [_, knn] = index.search(dataset, k, effective_efq);
                timer.stop();

                float recall = gt.recall(k, knn);
                float qps_val = dataset.num_ / timer.get();
                timer.reset();

                if (recall >= target)
                {
                    refined_upper = efq;
                    refined_recall = recall;
                    refined_qps = qps_val;
                    break;
                }
                else
                {
                    lower_efq = efq;
                    lower_recall = recall;
                    lower_qps = qps_val;
                }
            }

            optimal_efq = refined_upper;
            optimal_recall = refined_recall;
            optimal_qps = refined_qps;
            printf(" range target efq, max%d (recall=%.4f, qps=%.1f)\n",
                   optimal_efq, optimal_recall, optimal_qps, lower_efq, refined_upper);
        }

        if (achieved_recall)
        {
            *achieved_recall = optimal_recall;
        }
        if (achieved_qps)
        {
            *achieved_qps = optimal_qps;
        }

        return optimal_efq;
    }

    // efq recall( , for )
    float evaluate_recall_for_fixed_efq(IndexType &index,
                                        const DataSetWrapper<float> &dataset,
                                        const GroundTruth &gt,
                                        int k,
                                        int efq,
                                        float *achieved_qps = nullptr)
    {
        if (efq <= 0 || k <= 0)
        {
            if (achieved_qps)
            {
                *achieved_qps = 0.0f;
            }
            return 0.0f;
        }

        Timer timer;
        index.get_comparison_and_clear();
        timer.start();
        const int effective_efq = std::max(efq, k);
        auto [_, knn] = index.search(dataset, k, effective_efq);
        timer.stop();

        float recall = gt.recall(k, knn);
        if (achieved_qps)
        {
            float elapsed = timer.get();
            *achieved_qps = (elapsed > 0.0f) ? static_cast<float>(dataset.num_ / elapsed) : 0.0f;
        }

        return recall;
    }

    // efqparameters: k=10 recall >= 0.99 efq
    int find_optimal_efq_for_recall_99(IndexType &index,
                                       const DataSetWrapper<float> &query,
                                       const GroundTruth &gt,
                                       int R, int efc, int k = 10)
    {
        printf(" efqparameters: k=%d recall >= 0.99 efq \n", k);

        // 
        auto csv_data = CSVDataReader::read_csv(csv_path);
        vector<string> latest_row;
        int max_iter = -1;

        // skip ,
        for (int i = 1; i < csv_data.size(); i++)
        {
            const auto &row = csv_data[i];
            if (row.size() < 4)
                continue;

            try
            {
                int csv_R = stoi(row[0]);
                int csv_efc = stoi(row[1]);
                int csv_iter = stoi(row[2]);
                int csv_k = stoi(row[3]);

                if (csv_R == R && csv_efc == efc && csv_k == k && csv_iter > max_iter)
                {
                    max_iter = csv_iter;
                    latest_row = row;
                }
            }
            catch (const exception &)
            {
                continue;
            }
        }

        if (latest_row.empty())
        {
            printf(" efqparameters: k=%d recall >= 0.99 efq \n", k);
            return 500;
        }

        printf(" k=%d %d \n", k, max_iter);

        float achieved_recall = 0.0f;
        float achieved_qps = 0.0f;
        int optimal_efq = find_optimal_efq_for_target_recall(index, query, gt, k, 0.99,
                                                             5, 10, 2, 2000,
                                                             &achieved_recall, &achieved_qps);
        printf(" minefq=%d (recall=%.4f, qps=%.1f), [%d, %d]\n",
               k, optimal_efq, achieved_recall, achieved_qps);
        return optimal_efq;
    }

    // 
    float calculate_improvement_ratio(IndexType &index,
                                      const DataSetWrapper<float> &query,
                                      const GroundTruth &gt,
                                      int R, int efc, int k)
    {
        printf(" efqparameters: k=%d recall >= 0.99 efq \n", k);

        // recall-QPS
        vector<pair<float, float>> current_recall_qps = read_recent_recall_qps(R, efc, k);
        auto prev_record = read_previous_iteration_record(R, efc, k);
        vector<pair<float, float>> previous_recall_qps;
        if (!prev_record.empty())
        {
            previous_recall_qps = CSVDataReader::extract_recall_qps_pairs(prev_record);
        }

        // 
        if (current_recall_qps.size() < 2 || previous_recall_qps.size() < 2)
        {
            printf(" efqparameters: k=%d recall >= 0.99 efq \n", k);
            return 1.5f;
        }

        // recall , recall QPS
        vector<float> recall_ratios;
        printf(" efqparameters: k=%d recall >= 0.99 efq \n", k);

        for (const auto &current_pair : current_recall_qps)
        {
            float recall = current_pair.first;
            float current_qps = current_pair.second;

            // recall QPS
            auto it = find_if(previous_recall_qps.begin(), previous_recall_qps.end(),
                              [recall](const pair<float, float> &p)
                              {
                                  return abs(p.first - recall) < 1e-6;
                              });

            if (it != previous_recall_qps.end())
            {
                float previous_qps = it->second;
                if (previous_qps > 0)
                {
                    float ratio = current_qps / previous_qps;
                    recall_ratios.push_back(ratio);
                    printf(" efq : k=%d recall >= 0.99 efq %d (recall=%.4f, qps=%.1f)\n",
                           recall, current_qps, previous_qps, ratio);
                }
            }
        }

        // 
        if (recall_ratios.empty())
        {
            printf(" k=%d ...\n", k);
            return 1.0f;
        }

        // recall avg
        float avg_ratio = 0.0f;
        for (float ratio : recall_ratios)
        {
            avg_ratio += ratio;
        }
        avg_ratio /= recall_ratios.size();

        printf("k=%d : %.3f ( %d recall avg )\n", k, avg_ratio, (int)recall_ratios.size());

        // 
        if (avg_ratio <= 0)
        {
            printf("k=%d (%.3f), 1.0\n", k, avg_ratio);
            return 1.0f;
        }

        return avg_ratio;
    }
};

// ======================== PIGR9_2 ========================

class PIGR9_2Algorithm
{
private:
    IndexType &index;
    const DataSetWrapper<float> &base;
    const DataSetWrapper<float> &query;
    const GroundTruth &gt;
    PseudoGroundTruth &train_gt_;

    OptimizedEdgePruner pruner;
    SmartEdgeAdder smart_adder;
    PerformanceTester &tester;
    AdaptiveStrategyManager strategy_manager;
    Timer timer;

    struct Parameters
    {
        int R = 16;
        int efc = 96;
        int iternum = 20;
        int efq_maintree_prune = 1000;
        int efq_maintree_add = 500;
        int efq_train_gt = 2000;
        int efq_max = 1000;
        int efq_step = 30;
        int jump_max = 5;
        int repeat_max = 2;
        int flag_period = 1;
        float prune_ratio = 0.02;
        double prune_sample_ratio = 1.0;

        // parameters
        int total_batches = 4;
        int num_threads = 24;

        // global parameters
        float perturbation_percentage = 0.01f;
        float high_ratio_percentage = 0.02f;
        int max_connections_per_high_ratio = 2;
        int max_connections_per_node = 3;
        int global_connectivity_degree = 10;
        int unified_k = 10;
        float efq_prune_target_recall = 0.99f;
        int efq_train_gt_fixed = 2000;
        float global_connectivity_target_recall = 0.90f;
        int efq_global_connectivity = 800;

        int operation_level_scope = 1;

        string base_csv_path = "";
        string csv_path = "out/performance_PIGR9_2clear_sift_optimized.csv";
        // QPS kparameters
        int k1 = 1;
        int k2 = 10;
        int k3 = 50;
        int k4 = 100;
    } params;

public:
    PIGR9_2Algorithm(IndexType &idx,
                    const DataSetWrapper<float> &base_data,
                    const DataSetWrapper<float> &query_data,
                    const GroundTruth &ground_truth,
                    PseudoGroundTruth &train_ground_truth,
                    PerformanceTester &test, int threads = 24)
        : index(idx), base(base_data), query(query_data), gt(ground_truth), train_gt_(train_ground_truth),
          pruner(idx, threads), smart_adder(idx, threads), tester(test),
          strategy_manager(tester, "")
    {
        params.num_threads = threads;
    }

    // parameters
    void set_parameters(int R, int efc, int iternum, const string &base_csv_path)
    {
        params.R = R;
        params.efc = efc;
        params.iternum = iternum;
        params.base_csv_path = base_csv_path;

        printf(" parameters: R=%d, efc=%d, =%d\n", R, efc, iternum);
        printf("Base CSV path: %s\n", base_csv_path.c_str());
    }

    // edge parameters
    void set_edge_parameters(float prune_ratio, int jump_max, int repeat_max)
    {
        params.prune_ratio = prune_ratio;
        params.jump_max = jump_max;
        params.repeat_max = repeat_max;

        // parameters CSV
        size_t dot_pos = params.base_csv_path.find_last_of('.');
        string name_part = (dot_pos != string::npos) ? params.base_csv_path.substr(0, dot_pos) : params.base_csv_path;
        string ext_part = (dot_pos != string::npos) ? params.base_csv_path.substr(dot_pos) : ".csv";

        // edge 、Edge insertion Edge insertion parameters
        char param_suffix[100];
        sprintf(param_suffix, "_prune%.3f_jump%d_repeat%d", params.prune_ratio, params.jump_max, params.repeat_max);
        params.csv_path = name_part + param_suffix + ext_part;

        // CSV
        strategy_manager.update_csv_path(params.csv_path);

        printf(" Pruning : %.1f%%\n", prune_ratio * 100);
        printf(" Edge insertion : %d\n", jump_max);
        printf(" Edge insertion : %d\n", repeat_max);
        printf("Generated CSV path: %s\n", params.csv_path.c_str());
        printf(" : ( )\n");
    }

    void set_prune_sample_ratio(double ratio)
    {
        if (ratio <= 0.0 || ratio > 1.0)
        {
            printf("Warning: Pruning %.4f, 1.0( seed set). \n", ratio);
            params.prune_sample_ratio = 1.0;
        }
        else
        {
            params.prune_sample_ratio = ratio;
        }
        printf(" Pruning : %.2f%%\n", params.prune_sample_ratio * 100.0);
    }

    // efqparameters
    void set_efq_parameters(int efq_maintree_prune, int efq_maintree_add)
    {
        params.efq_maintree_prune = efq_maintree_prune;
        params.efq_maintree_add = efq_maintree_add;

        printf(" Pruningefq_maintree: %d\n", efq_maintree_prune);
        printf(" Edge insertionefq_maintree: %d\n", efq_maintree_add);
    }

    // targetrecall pseudo-GTfixedefq
    void set_dynamic_efq_targets(float prune_target_recall, int train_gt_fixed_efq)
    {
        if (prune_target_recall <= 0.0f || prune_target_recall > 1.0f)
        {
            printf("Warning: Pruningtargetrecall=%.4f, %.4f\n",
                   prune_target_recall, params.efq_prune_target_recall);
        }
        else
        {
            params.efq_prune_target_recall = prune_target_recall;
        }

        if (train_gt_fixed_efq > 0)
        {
            params.efq_train_gt_fixed = train_gt_fixed_efq;
            params.efq_train_gt = train_gt_fixed_efq;
        }
        else
        {
            printf("Warning: Pruningtargetrecall=%.4f, %.4f\n",
                   train_gt_fixed_efq, params.efq_train_gt_fixed);
        }

        printf(" Pruningtargetrecall: %.4f\n", params.efq_prune_target_recall);
        printf(" pseudo-GTfixedefq: %d\n", params.efq_train_gt_fixed);
    }

    // efqparameters
    void adjust_efq_parameters_dynamically(IndexType &index,
                                           const DataSetWrapper<float> &query,
                                           const GroundTruth &gt)
    {
        printf(" : ( )\n");

        float recall_unified = 0.0f;
        float qps_unified = 0.0f;
        const float target_recall_unified = params.efq_prune_target_recall;
        int efq_for_unified = strategy_manager.find_optimal_efq_for_target_recall(
            index, query, gt, params.unified_k, target_recall_unified, 5, 10, 2, 2000,
            &recall_unified, &qps_unified);

        int k_for_train_gt = params.unified_k;
        if (k_for_train_gt <= 0)
        {
            printf("Warning: k<=0, pseudo-GT k=%d\n", params.k3);
            k_for_train_gt = params.k3;
        }

        int efq_train_gt = params.efq_train_gt_fixed;
        if (efq_train_gt <= 0)
        {
            printf(" efqparameters...\n");
            efq_train_gt = 2000;
        }

        float qps_train_gt = 0.0f;
        float recall_train_gt = strategy_manager.evaluate_recall_for_fixed_efq(
            index, query, gt, k_for_train_gt, efq_train_gt, &qps_train_gt);

        int old_prune = params.efq_maintree_prune;
        int old_add = params.efq_maintree_add;
        int old_train_gt = params.efq_train_gt;

        params.efq_maintree_prune = efq_for_unified;
        params.efq_train_gt = efq_train_gt;
        params.efq_maintree_add = std::max(efq_for_unified * 2, efq_train_gt);

        printf(" efqparameters...\n");
        printf("Warning: pseudo-GTfixedefq=%d , %d\n",
               old_prune, params.efq_maintree_prune, params.unified_k, target_recall_unified,
               recall_unified, qps_unified);
        printf("Warning: pseudo-GTfixedefq=%d , %d\n",
               old_train_gt, params.efq_train_gt, k_for_train_gt, recall_train_gt, qps_train_gt);
        printf(" Edge insertionefq_maintree: %d -> %d\n", old_add, params.efq_maintree_add);

        // regenerate_train_ground_truth(params.efq_train_gt, params.unified_k);
    }

    // QPS kparameters
    void set_test_k_parameters(int k1, int k2, int k3, int k4)
    {
        params.k1 = k1;
        params.k2 = k2;
        params.k3 = k3;
        params.k4 = k4;

        printf(" QPS kparameters: k1=%d, k2=%d, k3=%d, k4=%d\n", k1, k2, k3, k4);
    }

    // parameters
    void set_batch_parameters(int total_batches, int num_threads)
    {
        params.total_batches = total_batches;
        params.num_threads = num_threads;

        printf(" parameters: total_batches=%d, num_threads=%d\n", total_batches, num_threads);
        printf(" parameters: total_batches=%d, num_threads=%d\n", total_batches, num_threads);
    }

    // global parameters
    void set_perturbation_parameters(float perturbation_percentage, int max_connections_per_node)
    {
        params.perturbation_percentage = perturbation_percentage;
        params.max_connections_per_node = max_connections_per_node;

        printf(" Seed-set pseudo-GT efq fixed: %d -> %d (k=%d, recall=%.4f, qps=%.1f)\n",
               perturbation_percentage * 100, max_connections_per_node);
        printf(" Seed-set pseudo-GT efq fixed: %d -> %d (k=%d, recall=%.4f, qps=%.1f)\n",
               perturbation_percentage * 100, perturbation_percentage * 100);
        printf(" node %d node\n", max_connections_per_node);
    }

    // node edge parameters
    void set_high_ratio_parameters(float high_ratio_percentage, int max_connections_per_high_ratio)
    {
        params.high_ratio_percentage = high_ratio_percentage;
        params.max_connections_per_high_ratio = max_connections_per_high_ratio;

        printf("efqparameters :\n");
        printf(" Seed-set pseudo-GT efq fixed: %d -> %d (k=%d, recall=%.4f, qps=%.1f)\n",
               high_ratio_percentage * 100, high_ratio_percentage * 100);
        printf(" global parameters: perturbation_percentage=%.3f%%, max_connections_per_node=%d\n",
               max_connections_per_high_ratio, max_connections_per_high_ratio);
    }

    void set_global_connectivity_degree(int degree)
    {
        set_unified_k(degree);
    }

    void set_global_connectivity_target_recall(float recall)
    {
        params.global_connectivity_target_recall = recall;
        printf(" global targetrecall: %.3f\n", recall);
    }

    void set_unified_k(int k)
    {
        params.unified_k = k;
        params.global_connectivity_degree = k;
        printf(" k : %d (Pruning/pseudo-GT/Edge insertion )\n", k);
    }

    void set_operation_level_scope(int level_scope)
    {
        int sanitized = level_scope <= 0 ? 1 : level_scope;
        params.operation_level_scope = sanitized;
        if (sanitized == 1)
        {
            printf(" node edgeparameters:\n");
        }
        else
        {
            printf(" level : 0level~%dlevel Pruning localEdge insertion. \n", sanitized - 1);
        }
    }

    double compute_recall_against_train_gt(int node_idx, const std::vector<int> &candidates, int k) const
    {
        const int *gt_row = train_gt_.row(node_idx);
        if (!gt_row)
        {
            return 0.0;
        }
        int limit = std::min(k, static_cast<int>(train_gt_.dim));
        if (limit <= 0)
        {
            return 0.0;
        }
        int hits = 0;
        for (int candidate : candidates)
        {
            for (int j = 0; j < limit; ++j)
            {
                if (gt_row[j] == candidate)
                {
                    ++hits;
                    break;
                }
            }
        }
        return static_cast<double>(hits) / static_cast<double>(limit);
    }

private:
    void regenerate_train_ground_truth(int efq, int k)
    {
        if (k <= 0)
        {
            printf(" level : 0level Pruning localEdge insertion. \n");
            return;
        }

        printf(" efq=%d seed set pseudo-GT (k=%d, node)...\n", efq, k);
        Timer t;
        t.start();
        const int effective_efq = std::max(efq, k);
        if (effective_efq != efq)
        {
            printf(" Note: efq=%d k=%d, pseudo-GT efq=%d\n", efq, k, effective_efq);
        }
        auto search_result = index.search(base, k, effective_efq);
        t.stop();

        auto &knn = search_result.second;
        size_t expected = static_cast<size_t>(base.num_) * static_cast<size_t>(k);
        if (knn.size() != expected)
        {
            printf(" Warning: knn (%zu) (%zu) , skip \n", knn.size(), expected);
        }
        else
        {
            train_gt_.reset(base.num_, k);
            const int *src = knn.data();
            for (size_t i = 0; i < base.num_; ++i)
            {
                const int *row_src = src + i * static_cast<size_t>(k);
                int *row_dst = train_gt_.row_mut(i);
                for (size_t j = 0; j < static_cast<size_t>(k); ++j)
                {
                    row_dst[j] = row_src[j];
                }
            }
            printf(" seed setpseudo-GT , %.2f \n", t.get());
        }

        search_result.first.clear();
        search_result.first.shrink_to_fit();
        search_result.second.clear();
        search_result.second.shrink_to_fit();
    }

    int strengthen_global_connectivity_from_train_gt(int topk)
    {
        if (topk <= 0)
        {
            return 0;
        }
        if (train_gt_.data.empty() || train_gt_.dim == 0)
        {
            printf("global skip: seed setGT \n");
            return 0;
        }

        size_t num_nodes = index.link_lists_.size();
        if (num_nodes == 0)
        {
            return 0;
        }

        int degree = std::min(topk, static_cast<int>(train_gt_.dim));
        if (degree <= 0)
        {
            printf("global skip: seed setGT \n");
            return 0;
        }

        float target_recall = params.global_connectivity_target_recall;
        if (target_recall < 0.0f)
            target_recall = 0.0f;
        if (target_recall > 1.0f)
            target_recall = 1.0f;
        float measured_recall = 0.0f;
        float measured_qps = 0.0f;
        if (target_recall > 0.0f)
        {
            params.efq_global_connectivity = strategy_manager.find_optimal_efq_for_target_recall(
                index, query, gt, degree, target_recall, 5, 10, 2, 2000,
                &measured_recall, &measured_qps);
        }
        else
        {
            measured_recall = 0.0f;
            measured_qps = 0.0f;
        }

        printf("global : targetrecall=%.3f, efq=%d ( recall=%.4f, qps=%.1f)\n",
               target_recall, params.efq_global_connectivity, measured_recall, measured_qps);

        const int effective_efq = std::max(params.efq_global_connectivity, degree);
        if (effective_efq != params.efq_global_connectivity)
        {
            printf("global : targetrecall=%.3f, efq=%d ( recall=%.4f, qps=%.1f)\n",
                   params.efq_global_connectivity, degree, effective_efq);
        }
        auto search_result = index.search(base, degree, effective_efq);
        const auto &knn = search_result.second;
        if (knn.size() != num_nodes * static_cast<size_t>(degree))
        {
            printf("global : targetrecall=%.3f, efq=%d ( recall=%.4f, qps=%.1f)\n",
                   knn.size(), num_nodes * static_cast<size_t>(degree));
            return 0;
        }

        std::vector<double> node_recalls(num_nodes, 0.0);
        std::vector<char> need_fix(num_nodes, 0);

        double recall_sum = 0.0;
        double min_recall = 1.0;
        double max_recall = 0.0;
        int below_target = 0;

        auto compute_node_recall = [&](size_t node_index, const int *search_ptr, const int *gt_ptr)
        {
            int hits = 0;
            for (int j = 0; j < degree; ++j)
            {
                int candidate = search_ptr[j];
                if (candidate < 0)
                    continue;
                for (int g = 0; g < degree; ++g)
                {
                    if (gt_ptr[g] == candidate)
                    {
                        ++hits;
                        break;
                    }
                }
            }
            double recall_value = (degree > 0) ? static_cast<double>(hits) / static_cast<double>(degree) : 0.0;
            node_recalls[node_index] = recall_value;
            recall_sum += recall_value;
            min_recall = std::min(min_recall, recall_value);
            max_recall = std::max(max_recall, recall_value);
            if (recall_value + 1e-6 < target_recall)
            {
                need_fix[node_index] = 1;
                below_target++;
            }
        };

        for (size_t node = 0; node < num_nodes; ++node)
        {
            const int *gt_row = train_gt_.row(node);
            const int *search_row = knn.data() + node * static_cast<size_t>(degree);
            if (!gt_row)
                continue;
            compute_node_recall(node, search_row, gt_row);
        }

        printf(" seed set recall@%d: avg=%.4f, =%.4f, =%.4f, target node=%d/%zu\n",
               degree, (num_nodes > 0) ? recall_sum / num_nodes : 0.0, min_recall, max_recall,
               below_target, num_nodes);

        std::vector<int> sample_nodes;
        sample_nodes.reserve(10);
        auto try_add_sample = [&](int node)
        {
            if (node < 0 || node >= static_cast<int>(num_nodes))
                return;
            if (sample_nodes.size() >= 10)
                return;
            if (std::find(sample_nodes.begin(), sample_nodes.end(), node) == sample_nodes.end())
            {
                sample_nodes.push_back(node);
            }
        };
        for (size_t node = 0; node < num_nodes && sample_nodes.size() < 10; ++node)
        {
            if (need_fix[node])
            {
                try_add_sample(static_cast<int>(node));
            }
        }
        for (size_t node = 0; node < num_nodes && sample_nodes.size() < 10; ++node)
        {
            try_add_sample(static_cast<int>(node));
        }
        std::vector<double> sample_recall_before;
        sample_recall_before.reserve(sample_nodes.size());
        for (int node : sample_nodes)
        {
            sample_recall_before.push_back(node_recalls[node]);
        }

        int added_edges = 0;
        int adjusted_nodes = 0;
        const int max_edges_per_node = 2;

        for (size_t node = 0; node < num_nodes; ++node)
        {
            if (!need_fix[node])
            {
                continue;
            }

            const int *gt_row = train_gt_.row(node);
            const int *search_row = knn.data() + node * static_cast<size_t>(degree);
            if (!gt_row || !search_row)
            {
                continue;
            }

            std::vector<int> current_results;
            current_results.reserve(degree);
            std::unordered_set<int> result_set;
            result_set.reserve(static_cast<size_t>(degree) * 2);
            for (int j = 0; j < degree; ++j)
            {
                int present = search_row[j];
                if (present < 0 || present >= static_cast<int>(num_nodes) || present == static_cast<int>(node))
                {
                    continue;
                }
                if (result_set.insert(present).second)
                {
                    current_results.push_back(present);
                }
            }

            if (current_results.empty())
            {
                continue;
            }

            // 
            std::vector<int> missing_neighbors;
            for (int j = 0; j < degree; ++j)
            {
                int gt_neighbor = gt_row[j];
                if (gt_neighbor < 0 || gt_neighbor >= static_cast<int>(num_nodes) || gt_neighbor == static_cast<int>(node))
                {
                    continue;
                }
                if (!result_set.count(gt_neighbor))
                {
                    missing_neighbors.push_back(gt_neighbor);
                }
            }

            if (missing_neighbors.empty())
            {
                continue;
            }

            // , Edge insertion, max_edges_per_node
            int edges_added_for_this_node = 0;
            for (int missing_id : missing_neighbors)
            {
                if (edges_added_for_this_node >= max_edges_per_node)
                {
                    break;
                }

                int best_source = -1;
                float best_distance = std::numeric_limits<float>::max();
                for (int present_id : current_results)
                {
                    if (present_id < 0 || present_id >= static_cast<int>(num_nodes) || present_id == missing_id)
                    {
                        continue;
                    }
                    float distance = metrics::euclidean(index.base_[present_id], index.base_[missing_id], index.base_.dim_);
                    if (distance < best_distance)
                    {
                        best_distance = distance;
                        best_source = present_id;
                    }
                }

                if (best_source < 0)
                {
                    continue;
                }

                int edges_added_now = 0;
                bool edge_added = false;
#pragma omp critical(nsg_neighbor_update)
                {
                    auto *source_neighbors = get_base_neighbors_mut(index, best_source);
                    auto *target_neighbors = get_base_neighbors_mut(index, missing_id);
                    if (source_neighbors && target_neighbors)
                    {
                        if (std::find(source_neighbors->begin(), source_neighbors->end(), missing_id) == source_neighbors->end())
                        {
                            source_neighbors->push_back(missing_id);
                            edge_added = true;
                            ++edges_added_now;
                        }

                        if (std::find(target_neighbors->begin(), target_neighbors->end(), best_source) == target_neighbors->end())
                        {
                            target_neighbors->push_back(best_source);
                            edge_added = true;
                            ++edges_added_now;
                        }
                    }
                }

                if (edge_added)
                {
                    added_edges += edges_added_now;
                    ++edges_added_for_this_node;
                }
            }

            if (edges_added_for_this_node > 0)
            {
                ++adjusted_nodes;
            }
        }

        search_result.first.clear();
        search_result.first.shrink_to_fit();
        search_result.second.clear();
        search_result.second.shrink_to_fit();

        printf(" recallnode global edge: node =%d, addedge=%d\n", adjusted_nodes, added_edges);

        int sample_k = std::min(degree, static_cast<int>(train_gt_.dim));
        if (!sample_nodes.empty() && params.efq_global_connectivity > 0 && sample_k > 0)
        {
            EdgeSearchTracker dummy_tracker;
            TrackedSearchExecutor sample_executor(index, dummy_tracker);
            std::vector<int> tmp_results;
            printf(" node recall (efq=%d, k=%d):\n", params.efq_global_connectivity, sample_k);
            for (size_t idx = 0; idx < sample_nodes.size(); ++idx)
            {
                int node = sample_nodes[idx];
                double before = sample_recall_before[idx];
                sample_executor.execute_search_baseline_fast(base[node], node,
                                                             params.efq_global_connectivity,
                                                             sample_k, tmp_results);
                double after = compute_recall_against_train_gt(node, tmp_results, sample_k);
                printf(" Node %d: Recall@%d %.4f -> %.4f (Δ=%.4f)\n",
                       node, sample_k, before, after, after - before);
            }
        }
        else if (!sample_nodes.empty() && params.efq_global_connectivity <= 0)
        {
            printf(" node recall skip: efq_global_connectivity<=0\n");
        }

        // HNSW parameters , NSG R_

        return added_edges;
    }

public:
    // 
    void run()
    {
        printf("\n=== ===\n");
        printf("\n=== ===\n");

        tester.init_csv(params.csv_path);

        printf("\n=== ===\n");
        auto [baseline_comparisons, baseline_aecr] = tester.calculate_total_comparisons(params.unified_k);
        printf(" avgdistance evaluations=%.2f, aecr=%.2f\n,recall=%.2f", baseline_comparisons, baseline_aecr, 0.99);

        const int test_ks[4] = {params.k1, params.k2, params.k3, params.k4};
        for (int k : test_ks)
        {
            tester.test_and_record(params.csv_path, params.R, params.efc, 0, k, 0,
                                   0, 0, params.efq_max, params.efq_step, baseline_aecr, baseline_comparisons, 0);
        }

        int total_cut_edges = 0;
        int total_truth_add_edges = 0;
        int total_tree_add_edges = 0;

        enum class OperationType
        {
            Pruning,
            Adding
        };
        OperationType current_operation = OperationType::Pruning;
        int switch_count = 0;

        for (int iter = 1; iter <= params.iternum; ++iter)
        {
            printf("\n=== %d/%d ===\n", iter, params.iternum);
            printf(" : %s\n", current_operation == OperationType::Pruning ? "Pruning" : "Edge insertion");

            adjust_efq_parameters_dynamically(index, query, gt);

            auto [pre_comparisons, pre_efq] = tester.calculate_total_comparisons(params.unified_k);
            printf(" avgdistance evaluations=%.2f, efq=%d\n", pre_comparisons, pre_efq);

            int operation_result = 0;
            int pruned_edges = 0;
            int local_added = 0;
            int global_added = 0;

            Timer opt_timer;
            opt_timer.reset();
            opt_timer.start();

            if (current_operation == OperationType::Pruning)
            {
                NewEdgePruningAlgorithm pruner_runner(index, base, pruner, train_gt_);
                pruner_runner.set_sampling_ratio(params.prune_sample_ratio);
                pruned_edges = pruner_runner.execute_new_pruning_algorithm(
                    params.prune_ratio,
                    params.efq_maintree_prune,
                    params.efq_maintree_prune,
                    params.unified_k,
                    params.total_batches,
                    params.operation_level_scope);
                total_cut_edges += pruned_edges;
                operation_result = pruned_edges;
                printf(" Pruning stage: removed edges=%d, cumulative removed=%d\n", pruned_edges, total_cut_edges);
            }
            else
            {
                regenerate_train_ground_truth(params.efq_train_gt, params.unified_k);

                global_added = strengthen_global_connectivity_from_train_gt(params.unified_k);
                if (global_added > 0)
                {
                    printf(" global : addedge=%d\n", global_added);
                }

                local_added = smart_adder.add_edges_by_search_tree(
                    base, params.efq_max, params.jump_max, params.repeat_max, params.unified_k, params.operation_level_scope);
                if (local_added > 0)
                {
                    printf(" localEdge insertion : addedge=%d\n", local_added);
                }

                operation_result = local_added + global_added;
                total_truth_add_edges += global_added;
                total_tree_add_edges += local_added;
                const int total_added_edges = total_truth_add_edges + total_tree_add_edges;
                printf(" Edge insertion stage: local=%d, global=%d, cumulative added (truth=%d, tree=%d, total=%d)\n",
                       local_added, global_added, total_truth_add_edges, total_tree_add_edges, total_added_edges);
            }
            opt_timer.stop();
            double opt_time = opt_timer.get();
            opt_timer.reset();

            auto [post_comparisons, post_acer] = tester.calculate_total_comparisons(params.unified_k);
            printf(" avgdistance evaluations=%.2f, acer=%.6f\n", post_comparisons, post_acer);

            for (int k : test_ks)
            {
                tester.test_and_record(params.csv_path, params.R, params.efc, iter, k,
                                       total_truth_add_edges, total_tree_add_edges, total_cut_edges, params.efq_max, params.efq_step, post_acer, post_comparisons, opt_time);
            }

            if (operation_result == 0)
            {
                printf("\n=== PIGR9_2 ===\n");
            }

            if (total_cut_edges > 100000000)
            {
                printf("\n=== PIGR9_2 ===\n");
                break;
            }

            if (iter < params.iternum)
            {
                current_operation = (current_operation == OperationType::Pruning)
                                        ? OperationType::Adding
                                        : OperationType::Pruning;
                ++switch_count;
            }
        }

        const int grand_total_add_edges = total_truth_add_edges + total_tree_add_edges;
        printf("\n=== PIGR9_2 ===\n");
        printf("total : removeedge =%d, truth_add=%d, tree_add=%d, totaladd=%d, =%d\n",
               total_cut_edges, total_truth_add_edges, total_tree_add_edges, grand_total_add_edges, switch_count);
    }
};




inline int run(const Config& cfg)
{
    // Basic sanity
    if (cfg.base_path.empty() || cfg.query_path.empty() || cfg.gt_path.empty() || cfg.index_path.empty())
    {
        std::cerr << "[pigr::refiner::hnsw] Missing required paths (base/query/gt/index)." << std::endl;
        return 2;
    }

    // Load dataset
    DataSetWrapper<float> base(cfg.base_path);
    DataSetWrapper<float> query(cfg.query_path);
    GroundTruth gt(cfg.gt_path);
    PseudoGroundTruth train_gt;

    // Load index
    IndexType index(base, cfg.index_path);

    // Create tester + algorithm
    Timer timer;
    PerformanceTester tester(index, base, query, gt, timer, cfg.threads);
    PIGR9_2Algorithm algorithm(index, base, query, gt, train_gt, tester, cfg.threads);

    // Params
    algorithm.set_parameters(cfg.param1, cfg.param2, cfg.param3, cfg.log_csv);

    algorithm.set_efq_parameters(cfg.efq_prune_init, cfg.efq_add_init);
    algorithm.set_dynamic_efq_targets(cfg.dynamic_target_recall, cfg.dynamic_max_efq);

    algorithm.set_test_k_parameters(cfg.test_k1, cfg.test_k2, cfg.test_k3, cfg.test_k4);
    algorithm.set_batch_parameters(cfg.total_batches, cfg.batch_threads);

    algorithm.set_unified_k(cfg.core_k);

    const int jump_max = (cfg.jump_max > 0) ? cfg.jump_max : int(2.0 * std::sqrt(double(cfg.core_k)));
    algorithm.set_edge_parameters(cfg.prune_ratio, jump_max, cfg.repeat_max);

    algorithm.set_prune_sample_ratio(cfg.prune_sample_ratio);
    algorithm.set_operation_level_scope(cfg.operation_level_scope);

    algorithm.run();
    return 0;
}


}}} // namespace pigr::refiner::hnsw
