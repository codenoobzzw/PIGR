#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>

namespace annstree
{

    class SearchNode
    {
    public:
        int node_id;                        // English note: original Chinese comment removed for anonymous release.
        float distance;                     // English note: original Chinese comment removed for anonymous release.
        SearchNode *parent;                 // English note: original Chinese comment removed for anonymous release.
        std::vector<SearchNode *> children; // English note: original Chinese comment removed for anonymous release.
        int depth;                          // English note: original Chinese comment removed for anonymous release.
        // English note: original Chinese comment removed for anonymous release.

        // English note: original Chinese comment removed for anonymous release.
        SearchNode(int id, float dist, int d,SearchNode *par = nullptr)
            : node_id(id), distance(dist), parent(par), depth(d) {}
    };

    class SearchTree
    {
    public:
        SearchNode *root; // English note: original Chinese comment removed for anonymous release.
        int tree_id;
        SearchTree(int root_id, float root_dist,int t_id)
        {
            tree_id=t_id;
            root = new SearchNode(root_id, root_dist, 0); // English note: original Chinese comment removed for anonymous release.
        }

        // English note: original Chinese comment removed for anonymous release.
        void add_child(SearchNode *parent_node, int child_id, float dist)
        {
            SearchNode *child_node = new SearchNode(child_id, dist, parent_node->depth + 1, parent_node);
            parent_node->children.push_back(child_node);
        }

        // English note: original Chinese comment removed for anonymous release.
        /* English note: original Chinese comment removed for anonymous release. */
    };

    class SearchTreeManager
    {
        public:
        // English note: original Chinese comment removed for anonymous release.
        std::unordered_map<int, SearchTree*> trees;
    
        // English note: original Chinese comment removed for anonymous release.
        int current_tree_id = 0;
    
        // English note: original Chinese comment removed for anonymous release.
        SearchTree* create_tree(int root_id,float root_dist) {
            // English note: original Chinese comment removed for anonymous release.
            SearchTree* new_tree = new SearchTree(root_id, root_dist,current_tree_id);
            // English note: original Chinese comment removed for anonymous release.
            // new_tree->root = new SearchNode(root_id, root_dist);
            trees[current_tree_id] = new_tree;
            current_tree_id++;  // English note: original Chinese comment removed for anonymous release.
            return new_tree;
        }
        
        // English note: original Chinese comment removed for anonymous release.
        void add_to_tree(int tree_id, int parent_node_id, int child_node_id, float dist) {
            SearchTree *tree = get_tree(tree_id);
            if (tree) {
                SearchNode *parent_node = find_node_in_tree(tree->root, parent_node_id);
                if (parent_node) {
                    tree->add_child(parent_node, child_node_id, dist);
                }
            }
        }
    
        // English note: original Chinese comment removed for anonymous release.
        SearchTree* get_tree(int tree_id) {
            auto it = trees.find(tree_id);
            if (it != trees.end()) {
                return it->second;
            }
            return nullptr;
        }
    
        // English note: original Chinese comment removed for anonymous release.
        void delete_tree(int tree_id) {
            auto it = trees.find(tree_id);
            if (it != trees.end()) {
                delete it->second;
                trees.erase(it);
            }
        }
        SearchNode* find_node_in_tree(SearchNode* node, int target_id) {
            // English note: original Chinese comment removed for anonymous release.
            if (node->node_id == target_id) {
                return node;
            }
        
            // English note: original Chinese comment removed for anonymous release.
            for (auto child : node->children) {
                SearchNode* found = find_node_in_tree(child, target_id);
                if (found) {
                    return found;
                }
            }
        
            // English note: original Chinese comment removed for anonymous release.
            return nullptr;
        }
        // English note: original Chinese comment removed for anonymous release.
        /*
        std::vector<int> get_path(int root_id, int target_node_id)
        {
            SearchTree *tree = get_tree(root_id);
            if (tree)
            {
                SearchNode *target_node = find_node_in_tree(tree->root, target_node_id);
                if (target_node)
                {
                    return tree->get_path_to_root(target_node);
                }
            }
            return {};
        }*/
    };
}