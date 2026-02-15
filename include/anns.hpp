#pragma once

#include <numeric>
#include <vector>
#include <utility>
#include <functional>
#include <mutex>
#include <utils/binary.hpp>
#include <cstring>
#include <unordered_set>
#include <shared_mutex>
#include <thread>
#define k_subadj 10

struct PairHash
{
  size_t operator()(const std::pair<int, int> &p) const
  {
    // English note: original Chinese comment removed for anonymous release.
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};

struct subadj
{
  int data[k_subadj][k_subadj]; // English note: original Chinese comment removed for anonymous release.
};

namespace anns
{
  using knn_t = std::vector<int>;
  using dis_t = std::vector<float>;
  using res_t = std::pair<dis_t, knn_t>;

#define MAGIC_ID 0x3f3f3f3f
#define MAGIC_DIST std::numeric_limits<float>::max()
#define MAGIC_DIMENSION 4096

#define EPSILON std::numeric_limits<float>::denorm_min()

  inline int DEFAULT_HASH(int id)
  {
    return id;
  }

  /// @brief  {base pointer, num, dimension}
  /// @tparam data_t
  template <typename data_t>
  struct DataSet
  {
    const data_t *data_{nullptr};
    size_t num_{0};
    size_t dim_{0};
    std::function<int(int)> hash_{DEFAULT_HASH};

    inline const data_t *access(int id) const
    {
      return data_ + hash_(id) * dim_;
    }

    inline const data_t *operator[](int id) const
    {
      return data_ + hash_(id) * dim_;
    }
  };

  template <typename data_t>
  struct DataSetWrapper : public DataSet<data_t>
  {
    using DataSet<data_t>::data_;
    using DataSet<data_t>::num_;
    using DataSet<data_t>::dim_;
    using DataSet<data_t>::hash_;
    std::vector<data_t> base_;

    DataSetWrapper(const std::string &fname)
    {
      load(fname);
    }

    void load(const std::string &filename, bool bin = false) noexcept
    {
      if (bin)
        std::tie(num_, dim_) = utils::load_from_file_bin(base_, filename);
      else
        std::tie(num_, dim_) = utils::load_from_file(base_, filename);
      data_ = base_.data();
    }
  };

  // tree operate-------------------------------------------------------------------------

  class SearchNode
  {
  public:
    int node_id;                        // English note: original Chinese comment removed for anonymous release.
    float distance;                     // English note: original Chinese comment removed for anonymous release.
    SearchNode *parent;                 // English note: original Chinese comment removed for anonymous release.
    std::vector<SearchNode *> children; // English note: original Chinese comment removed for anonymous release.
    int depth;                          // English note: original Chinese comment removed for anonymous release.
    int chosen = 0;                     // English note: original Chinese comment removed for anonymous release.
    // int correct =0;                     // 0:
    // English note: original Chinese comment removed for anonymous release.

    // English note: original Chinese comment removed for anonymous release.
    SearchNode(int id, float dist, int d, SearchNode *par = nullptr)
        : node_id(id), distance(dist), parent(par), depth(d) {}
    // English note: original Chinese comment removed for anonymous release.
    ~SearchNode()
    {
      for (auto child : children)
      {
        delete child;
      }
    }
  };

  class SearchTree
  {
  public:
    SearchNode *root; // English note: original Chinese comment removed for anonymous release.
    int tree_id;
    int count = 0;
    SearchTree(int root_id, float root_dist, int t_id)
    {
      tree_id = t_id;
      root = new SearchNode(root_id, root_dist, 0); // English note: original Chinese comment removed for anonymous release.
    }
    // English note: original Chinese comment removed for anonymous release.
    ~SearchTree()
    {
      delete root;
    }
    // English note: original Chinese comment removed for anonymous release.
    void add_child(SearchNode *parent_node, int child_id, float dist)
    {
      SearchNode *child_node = new SearchNode(child_id, dist, parent_node->depth + 1, parent_node);
      parent_node->children.push_back(child_node);
    }

    void print_tree(SearchNode *node, int depth = 0)
    {

      printf("b");
      if (node == nullptr)
      {
        return;
      }
      // English note: original Chinese comment removed for anonymous release.

      std::cout << std::string(depth * 2, ' ') << "Node ID: " << node->node_id
                << " (Depth: " << node->depth << ", Distance: " << node->distance << ",chosen" << node->chosen << ",count" << (count++) << ")\n";

      // English note: original Chinese comment removed for anonymous release.

      for (auto child : node->children)
      {
        if (depth > 6)
        {
          break;
        }
        print_tree(child, depth + 1); // English note: original Chinese comment removed for anonymous release.
      }
    }

    // English note: original Chinese comment removed for anonymous release.
    void print_tree()
    {
      // printf("aaa");
      print_tree(root); // English note: original Chinese comment removed for anonymous release.
    }
  };

  class SearchTreeManager
  {
  private:
    std::mutex trees_mutex;

    std::shared_mutex trees_m;
    std::mutex node_mutex; // English note: original Chinese comment removed for anonymous release.
    std::mutex edge_mutex; // English note: original Chinese comment removed for anonymous release.
                           // std::mutex trees_mutex2;

  public:
    // English note: original Chinese comment removed for anonymous release.
    std::unordered_map<int, SearchTree *> trees;

    // English note: original Chinese comment removed for anonymous release.
    int current_tree_id = 0;
    std::vector<std::vector<int>> c2node; // chosen=2
    ~SearchTreeManager()
    {
      for (auto &pair : trees)
      {
        delete pair.second;
      }
      trees.clear();
    }
    // English note: original Chinese comment removed for anonymous release.
    SearchTree *create_tree(int root_id, float root_dist, int tree_id)
    {
      // English note: original Chinese comment removed for anonymous release.

      SearchTree *new_tree = new SearchTree(root_id, root_dist, tree_id);

      // std::unique_lock<std::mutex> lock(trees_mutex);
      trees[tree_id] = new_tree;

      return new_tree;
    }

    // English note: original Chinese comment removed for anonymous release.
    void add_to_tree(int tree_id, int parent_node_id, int child_node_id, float dist)
    {

      // std::unique_lock<std::mutex> lock(trees_mutex);
      SearchTree *tree = get_tree(tree_id);

      // English note: original Chinese comment removed for anonymous release.
      if (tree)
      {
        SearchNode *parent_node = find_node_in_tree(tree->root, parent_node_id);
        if (parent_node)
        {
          tree->add_child(parent_node, child_node_id, dist);
          // printf("www");
        }
      }
    }

    void add_to_tree_multithread(SearchTree *tree, int parent_node_id, int child_node_id, float dist)
    {

      // std::unique_lock<std::mutex> lock(trees_mutex);
      // SearchTree *tree = get_tree(tree_id);
      // English note: original Chinese comment removed for anonymous release.
      if (tree)
      {
        SearchNode *parent_node = find_node_in_tree(tree->root, parent_node_id);
        if (parent_node)
        {
          tree->add_child(parent_node, child_node_id, dist);
          // printf("www");
        }
      }
    }

    // English note: original Chinese comment removed for anonymous release.
    SearchTree *get_tree(int tree_id)
    {
      // std::unique_lock<std::mutex> lock(trees_mutex);
      auto it = trees.find(tree_id);
      if (it != trees.end())
      {
        return it->second;
      }

      else
      {
        printf("tree not found\n");
      }
      return nullptr;
    }

    // English note: original Chinese comment removed for anonymous release.
    void delete_tree(int tree_id)
    {
      auto it = trees.find(tree_id);
      if (it != trees.end())
      {
        delete it->second;
        trees.erase(it);
      }
    }
    SearchNode *find_node_in_tree(SearchNode *node, int target_id)
    {
      // English note: original Chinese comment removed for anonymous release.
      if (node->node_id == target_id)
      {
        return node;
      }

      // English note: original Chinese comment removed for anonymous release.
      for (auto child : node->children)
      {
        SearchNode *found = find_node_in_tree(child, target_id);
        if (found)
        {
          return found;
        }
      }

      // English note: original Chinese comment removed for anonymous release.
      return nullptr;
    }
    int count_nodes(SearchNode *node)
    {
      if (node == nullptr)
      {
        return 0;
      }

      int count = 1; // English note: original Chinese comment removed for anonymous release.
      for (auto child : node->children)
      {
        count += count_nodes(child); // English note: original Chinese comment removed for anonymous release.
      }
      return count;
    }

    // English note: original Chinese comment removed for anonymous release.
    void save_trees_to_file(const std::string &filename)
    {
      std::ofstream file(filename, std::ios::binary);
      if (!file.is_open())
      {
        std::cerr << "Failed to open file for writing.\n";
        return;
      }

      // English note: original Chinese comment removed for anonymous release.
      for (const auto &pair : trees)
      {
        int tree_id = pair.first;
        SearchTree *tree = pair.second;

        // English note: original Chinese comment removed for anonymous release.
        int num_nodes = count_nodes(tree->root);                             // English note: original Chinese comment removed for anonymous release.
        file.write(reinterpret_cast<char *>(&num_nodes), sizeof(num_nodes)); // English note: original Chinese comment removed for anonymous release.

        save_tree_to_file(file, tree->root, 0); // English note: original Chinese comment removed for anonymous release.
      }

      file.close();
      std::cout << "Trees saved to binary file successfully.\n";
    }

    // English note: original Chinese comment removed for anonymous release.
    void save_tree_to_file(std::ofstream &file, SearchNode *node, int parent_id)
    {
      if (node == nullptr)
      {
        return;
      }

      // English note: original Chinese comment removed for anonymous release.
      file.write(reinterpret_cast<char *>(&node->node_id), sizeof(node->node_id)); // English note: original Chinese comment removed for anonymous release.
      file.write(reinterpret_cast<char *>(&parent_id), sizeof(parent_id));         // English note: original Chinese comment removed for anonymous release.
      file.write(reinterpret_cast<char *>(&node->depth), sizeof(node->depth));     // English note: original Chinese comment removed for anonymous release.
      file.write(reinterpret_cast<char *>(&node->chosen), sizeof(node->chosen));   // English note: original Chinese comment removed for anonymous release.

      file.write(reinterpret_cast<char *>(&node->distance), sizeof(node->distance)); // English note: original Chinese comment removed for anonymous release.

      // English note: original Chinese comment removed for anonymous release.
      for (auto child : node->children)
      {
        save_tree_to_file(file, child, node->node_id); // English note: original Chinese comment removed for anonymous release.
      }
    }

    void extract_main_path_to_chosen_2(int tree_id)
    {
      SearchTree *tree = get_tree(tree_id);
      SearchNode *chosen_node = find_chosen_node(tree->root, 2); // English note: original Chinese comment removed for anonymous release.
      if (chosen_node)
      {
        // English note: original Chinese comment removed for anonymous release.
        std::vector<SearchNode *> path = get_path_to_node(tree->root, chosen_node);
        // English note: original Chinese comment removed for anonymous release.
        delete_other_nodes(tree->root, path);
      }
      else
      {
        delete_subtree(tree->root); // English note: original Chinese comment removed for anonymous release.
        tree->root = nullptr;       // English note: original Chinese comment removed for anonymous release.
      }
    }
    // English note: original Chinese comment removed for anonymous release.

    void mainpath_extract_info(std::vector<int> &node_visit_count, std::unordered_map<std::pair<int, int>, int, PairHash> &edge_visits, int tree_id)
    {
      SearchTree *tree = get_tree(tree_id);
      SearchNode *chosen_node = find_chosen_node(tree->root, 2); // English note: original Chinese comment removed for anonymous release.
      if (chosen_node)
      {
        std::vector<SearchNode *> path = get_path_to_node(tree->root, chosen_node);
        // English note: original Chinese comment removed for anonymous release.
        for (auto node : path)
        {
          node_visit_count[node->node_id]++;
          if (node->parent != nullptr)
          {
            edge_visits[{node->parent->node_id, node->node_id}]++;
          }
        }
      }
    }

    void mainpath_extract_info_observe(std::vector<int> &node_visit_count, std::unordered_map<std::pair<int, int>, int, PairHash> &edge_visits, int tree_id, std::vector<int> &depth)
    {
      SearchTree *tree = get_tree(tree_id);
      SearchNode *chosen_node = find_chosen_node(tree->root, 2); // English note: original Chinese comment removed for anonymous release.
      if (chosen_node)
      {
        std::vector<SearchNode *> path = get_path_to_node(tree->root, chosen_node);
        // English note: original Chinese comment removed for anonymous release.
        for (auto node : path)
        {
          node_visit_count[node->node_id]++;
          if (node->parent != nullptr)
          {
            edge_visits[{node->parent->node_id, node->node_id}]++;
          }
        }

        depth[tree_id] = chosen_node->depth;
      }
    }
    /* English note: original Chinese comment removed for anonymous release. */
    void mainpath_shorten_tree(std::vector<int> &node_visit_count, std::unordered_map<std::pair<int, int>, int, PairHash> &edge_visits, int tree_id, std::vector<std::vector<int>> &index, int &count)
    {
      // English note: original Chinese comment removed for anonymous release.
      SearchTree *tree = get_tree(tree_id);
      if (!tree || !tree->root)
        return;

      // English note: original Chinese comment removed for anonymous release.
      SearchNode *chosen2 = find_chosen_node(tree->root, 2);
      if (!chosen2)
        return;
      std::vector<SearchNode *> main_path = get_path_to_node(tree->root, chosen2);
      // English note: original Chinese comment removed for anonymous release.
      if (main_path.size() <= (7 + 1))
        return;

      // English note: original Chinese comment removed for anonymous release.
      SearchNode *candidate = nullptr;
      for (auto node : main_path)
      {
        if (node_visit_count[node->node_id] < index[node->node_id].size())
        {
          candidate = node;
          break;
        }
      }
      if (!candidate)
        return;

      // English note: original Chinese comment removed for anonymous release.
      SearchNode *deepest_node = chosen2;

      index[candidate->node_id].push_back(deepest_node->node_id);

      // edge_visits[{candidate->node_id, deepest_node->node_id}]++;

      count++;
    }

    void allpath_extract_info(
        std::vector<int> &node_visit_count,
        std::unordered_map<std::pair<int, int>, int, PairHash> &edge_visits,
        int tree_id)
    {
      /*
      SearchTree *tree = [&]
      {
        std::unique_lock<std::shared_mutex> lock(trees_m);
        return get_tree(tree_id);
      }();*/

      SearchTree *tree = get_tree(tree_id);

      if (!tree || !tree->root)
        return;

      std::vector<std::pair<int, int>> edges;
      std::unordered_set<int> non_leaf_nodes;

      // English note: original Chinese comment removed for anonymous release.
      std::unordered_map<int, int> local_node_visit_count;
      std::unordered_map<std::pair<int, int>, int, PairHash> local_edge_visits;

      std::function<void(SearchNode *)> dfs = [&](SearchNode *node)
      {
        if (!node)
          return;
        if (!node->children.empty())
        {
          local_node_visit_count[node->node_id]++; // English note: original Chinese comment removed for anonymous release.
          non_leaf_nodes.insert(node->node_id);
        }
        for (auto child : node->children)
        {
          edges.emplace_back(node->node_id, child->node_id);
          dfs(child);
        }
      };

      dfs(tree->root);

      for (const auto &edge : edges)
      {
        int child_id = edge.second;
        if (non_leaf_nodes.count(child_id))
        {
          local_edge_visits[edge]++;
        }
      }

      // English note: original Chinese comment removed for anonymous release.
      {
        std::unique_lock<std::mutex> lock(node_mutex);
        for (const auto &entry : local_node_visit_count)
        {
          node_visit_count[entry.first] += entry.second;
        }
      }
      // English note: original Chinese comment removed for anonymous release.
      {
        std::unique_lock<std::mutex> lock(edge_mutex);
        for (const auto &entry : local_edge_visits)
        {
          edge_visits[entry.first] += entry.second;
        }
      }
    }

    SearchNode *find_chosen_node(SearchNode *node, int target)
    {
      if (node == nullptr)
      {
        return nullptr;
      }

      if (node->chosen == target)
      {
        return node;
      }

      for (auto child : node->children)
      {
        SearchNode *found = find_chosen_node(child, target);
        if (found)
        {
          return found;
        }
      }

      return nullptr;
    }

    // English note: original Chinese comment removed for anonymous release.
    std::vector<SearchNode *> get_path_to_node(SearchNode *current, SearchNode *target)
    {
      if (current == nullptr)
      {
        return {};
      }

      if (current == target)
      {
        return {current}; // English note: original Chinese comment removed for anonymous release.
      }

      for (auto child : current->children)
      {
        std::vector<SearchNode *> path = get_path_to_node(child, target);
        if (!path.empty())
        {
          path.insert(path.begin(), current); // English note: original Chinese comment removed for anonymous release.
          return path;
        }
      }

      return {}; // English note: original Chinese comment removed for anonymous release.
    }

    // English note: original Chinese comment removed for anonymous release.
    void delete_other_nodes(SearchNode *current, const std::vector<SearchNode *> &path)
    {
      if (current == nullptr)
        return;

      // English note: original Chinese comment removed for anonymous release.
      for (auto it = current->children.begin(); it != current->children.end();)
      {
        SearchNode *child = *it;
        // English note: original Chinese comment removed for anonymous release.
        if (std::find(path.begin(), path.end(), child) == path.end())
        {
          delete_subtree(child);            // English note: original Chinese comment removed for anonymous release.
          it = current->children.erase(it); // English note: original Chinese comment removed for anonymous release.
        }
        else
        {
          // English note: original Chinese comment removed for anonymous release.
          delete_other_nodes(child, path);
          ++it;
        }
      }
    }

    // English note: original Chinese comment removed for anonymous release.
    void delete_subtree(SearchNode *node)
    {
      if (node == nullptr)
        return;
      for (auto child : node->children)
      {
        delete_subtree(child);
      }
      delete node;
    }

    //
    subadj get_subadj(int tree_id, int num, int jump_max, std::vector<std::vector<int>> &neighbors_, int &add_count)
    {
      subadj result;
      // English note: original Chinese comment removed for anonymous release.
      SearchTree *tree = get_tree(tree_id);
      if (!tree || !tree->root)
        return result;

      // English note: original Chinese comment removed for anonymous release.
      std::vector<std::pair<int, float>> node_dist_list; // {node_id, distance}
      std::function<void(SearchNode *)> dfs = [&](SearchNode *node)
      {
        if (!node)
          return;
        node_dist_list.emplace_back(node->node_id, node->distance);
        for (auto child : node->children)
          dfs(child);
      };
      dfs(tree->root);

      // English note: original Chinese comment removed for anonymous release.
      std::sort(node_dist_list.begin(), node_dist_list.end(),
                [](const auto &a, const auto &b)
                { return a.second < b.second; });
      if (node_dist_list.size() > num)
        node_dist_list.resize(num);

      // English note: original Chinese comment removed for anonymous release.
      std::vector<int> selected_ids;
      std::unordered_map<int, int> id2idx;
      for (int i = 0; i < node_dist_list.size(); ++i)
      {
        selected_ids.push_back(node_dist_list[i].first);
        id2idx[node_dist_list[i].first] = i;
      }

      // English note: original Chinese comment removed for anonymous release.
      for (int i = 0; i < num; ++i)
      {
        for (int j = 0; j < num; ++j)
        {
          result.data[i][j] = 0; // English note: original Chinese comment removed for anonymous release.
        }
      }
      for (int i = 0; i < selected_ids.size(); ++i)
      {
        int u = selected_ids[i];
        for (int v : neighbors_[u])
        {
          auto it = id2idx.find(v);
          if (it != id2idx.end())
          {
            int j = it->second;
            result.data[i][j] = 1; // English note: original Chinese comment removed for anonymous release.
          }
        }
      }

      // English note: original Chinese comment removed for anonymous release.
      /* English note: original Chinese comment removed for anonymous release. */

      {

        // English note: original Chinese comment removed for anonymous release.
        int A0[k_subadj][k_subadj] = {0};
        std::memcpy(A0, result.data, sizeof(A0));

        // English note: original Chinese comment removed for anonymous release.
        for (int repeat = 0; repeat < 5; repeat++)
        {
          // English note: original Chinese comment removed for anonymous release.
          int Am1[k_subadj][k_subadj] = {0};
          std::memcpy(Am1, A0, sizeof(Am1));
          for (int jump = 1; jump < jump_max; ++jump)
          {
            int temp[k_subadj][k_subadj] = {0};
            for (int i = 0; i < num; ++i)
            {
              for (int j = 0; j < num; ++j)
              {
                for (int k = 0; k < num; ++k)
                {
                  if (Am1[i][j] == 1 || (Am1[i][k] && A0[k][j]))
                  {
                    temp[i][j] = 1;
                    break;
                  }
                }
              }
            }
            std::memcpy(Am1, temp, sizeof(Am1));
          }

          // English note: original Chinese comment removed for anonymous release.
          int max_zero_count = -1;
          int target_row = -1;
          for (int i = 0; i < num; ++i)
          {
            int zero_count = 0;
            for (int j = 0; j < num; ++j)
            {
              if (Am1[i][j] == 0)
                zero_count++;
            }
            if (zero_count > max_zero_count)
            {
              max_zero_count = zero_count;
              target_row = i;
            }
          }
          if (target_row == -1 || max_zero_count == 0)
            break; // English note: original Chinese comment removed for anonymous release.

          // English note: original Chinese comment removed for anonymous release.
          int Am2[k_subadj][k_subadj];
          std::memcpy(Am2, A0, sizeof(Am2));
          for (int jump = 1; jump < jump_max - 1; ++jump)
          {
            int temp[k_subadj][k_subadj] = {0};
            for (int i = 0; i < num; ++i)
            {
              for (int j = 0; j < num; ++j)
              {
                for (int k = 0; k < num; ++k)
                {
                  if (Am2[i][j] == 1 || Am2[i][k] && A0[k][j])
                  {
                    temp[i][j] = 1;
                    break;
                  }
                }
              }
            }
            std::memcpy(Am2, temp, sizeof(Am2));
          }

          // English note: original Chinese comment removed for anonymous release.
          std::vector<int> candidate_cols;
          for (int j = 0; j < num; ++j)
          {
            if (Am1[target_row][j] == 0)
              candidate_cols.push_back(j);
          }

          // English note: original Chinese comment removed for anonymous release.
          int best_col = -1;
          int max_conn = -1;
          for (int col : candidate_cols)
          {
            int conn = 0;
            for (int k = 0; k < num; ++k)
            {
              if (Am2[col][k])
                conn++;
            }
            if (conn > max_conn)
            {
              max_conn = conn;
              best_col = col;
            }
          }

          // English note: original Chinese comment removed for anonymous release.
          if (best_col != -1)
          {
            A0[target_row][best_col] = 1;
            result.data[target_row][best_col] = 1;
            int u = selected_ids[target_row];
            int v = selected_ids[best_col];
            // English note: original Chinese comment removed for anonymous release.
            if (std::find(neighbors_[u].begin(), neighbors_[u].end(), v) == neighbors_[u].end())
            {
              // English note: original Chinese comment removed for anonymous release.
              neighbors_[u].emplace_back(v);
              add_count++;
            }
          }
          else
          {
            break; // English note: original Chinese comment removed for anonymous release.
          }
        }
      }

      return result;
    }
  };

}