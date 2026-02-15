#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <cmath>
// #include"matplotlibcpp.h"
#include <stack>
#include <queue>
#include <algorithm>
#include <fstream>
#include <utils/timer.hpp>
#include <omp.h>
#include <filesystem>
#include <distance.hpp>
#include <unordered_set>
using namespace std;
using namespace anns::utils;
using namespace anns::metrics;
namespace anns
{
	namespace judge
	{
		using id_t = int;
		template <typename data_t>
		class Judge_all
		{
			// English note: original Chinese comment removed for anonymous release.
			// English note: original Chinese comment removed for anonymous release.
			// English note: original Chinese comment removed for anonymous release.
		public:
			std::vector<std::vector<id_t>> adjlist_;
			std::vector<std::vector<float>> pointcoordinates_;
			std::vector<int> in_degrees_;
			std::vector<int> out_degrees_;
			std::vector<std::tuple<int, int, float>> distances_; // English note: original Chinese comment removed for anonymous release.
			std::vector<double> between_centrality_;
			std::vector<float> effectivesize_;
			std::vector<float> wfCentrality_;
			double averagePathLength_;
			int diameter_;
			int D_;
			size_t N_;

			// English note: original Chinese comment removed for anonymous release.
			double dampingFactor_{0.85};
			int maxIterations_{100};
			double tolerance_{1e-7};
			std::vector<double> pagerank_;
			// English note: original Chinese comment removed for anonymous release.
			std::vector<double> means_{0};
			std::vector<double> variances_{0};
			std::vector<double> skewnesses_{0};
			std::vector<double> kurtoses_{0};
			// DBSCAN
			double eps_{200};
			std::vector<id_t> labels_;
			int minPts_{300};

		public:
			struct pair_hash
			{
				template <typename T1, typename T2>
				std::size_t operator()(const std::pair<T1, T2> &p) const
				{
					auto h1 = std::hash<T1>{}(p.first);
					auto h2 = std::hash<T2>{}(p.second);
					return h1 ^ (h2 << 1);
				}
			};

			void calculateDegrees() // English note: original Chinese comment removed for anonymous release.
			{

				int n = adjlist_.size();
				out_degrees_.assign(n, 0);
				in_degrees_.assign(n, 0);

				// English note: original Chinese comment removed for anonymous release.
				for (int u = 0; u < n; ++u)
				{
					out_degrees_[u] = adjlist_[u].size();
					for (int v : adjlist_[u])
					{
						in_degrees_[v]++;
					}
				}
			}
			void calculateDistributions(
				std::unordered_map<int, double> &ax,							 // English note: original Chinese comment removed for anonymous release.
				std::unordered_map<int, double> &by,							 // English note: original Chinese comment removed for anonymous release.
				std::unordered_map<std::pair<int, int>, double, pair_hash> &exy) // English note: original Chinese comment removed for anonymous release.
			{
				size_t total_edges = 0;
				std::unordered_map<int, int> ax_count;
				std::unordered_map<int, int> by_count;
				std::unordered_map<std::pair<int, int>, int, pair_hash> exy_count;

				// English note: original Chinese comment removed for anonymous release.
				for (id_t u = 0; u < adjlist_.size(); u++)
				{
					int out_degree = out_degrees_[u]; // English note: original Chinese comment removed for anonymous release.
					for (id_t v : adjlist_[u])
					{									// English note: original Chinese comment removed for anonymous release.
						int in_degree = in_degrees_[v]; // English note: original Chinese comment removed for anonymous release.
						// English note: original Chinese comment removed for anonymous release.
						exy_count[{out_degree, in_degree}]++;
						ax_count[out_degree]++;
						by_count[in_degree]++;
						total_edges++; // English note: original Chinese comment removed for anonymous release.
					}
				}

				// English note: original Chinese comment removed for anonymous release.
				for (const auto &entry : exy_count)
				{
					// English note: original Chinese comment removed for anonymous release.
					exy[entry.first] = static_cast<double>(entry.second) / total_edges;
				}
				for (const auto &entry : ax_count)
				{
					ax[entry.first] = static_cast<double>(entry.second) / total_edges;
				}
				for (const auto &entry : by_count)
				{
					by[entry.first] = static_cast<double>(entry.second) / total_edges;
				}
			}

			double CalculateAssortativity_core(const std::unordered_map<int, double> &ax,							  // English note: original Chinese comment removed for anonymous release.
											   const std::unordered_map<int, double> &by,							  // English note: original Chinese comment removed for anonymous release.
											   const std::unordered_map<std::pair<int, int>, double, pair_hash> &exy) // English note: original Chinese comment removed for anonymous release.
			{
				double numerator = 0.0;
				double mean_a = 0.0, mean_b = 0.0;
				double variance_a = 0.0, variance_b = 0.0;

				// English note: original Chinese comment removed for anonymous release.
				for (const auto &entry : ax)
				{
					mean_a += entry.first * entry.second; // English note: original Chinese comment removed for anonymous release.
				}
				for (const auto &entry : by)
				{
					mean_b += entry.first * entry.second; // English note: original Chinese comment removed for anonymous release.
				}

				// English note: original Chinese comment removed for anonymous release.
				for (const auto &entry : exy)
				{
					int x = entry.first.first;
					int y = entry.first.second;
					double exy_value = entry.second;

					numerator += x * y * (exy_value - ax.at(x) * by.at(y)); // English note: original Chinese comment removed for anonymous release.
					variance_a += (x - mean_a) * (x - mean_a) * ax.at(x);	// English note: original Chinese comment removed for anonymous release.
					variance_b += (y - mean_b) * (y - mean_b) * by.at(y);	// English note: original Chinese comment removed for anonymous release.
				}

				double denominator = std::sqrt(variance_a) * std::sqrt(variance_b);
				return (denominator != 0) ? (numerator / denominator) : 0;
			}
			// English note: original Chinese comment removed for anonymous release.
			size_t countTriangles(id_t id, int degree)
			{
				// English note: original Chinese comment removed for anonymous release.

				size_t trianglecount = 0;
				for (id_t u : adjlist_[id])
				{
					for (id_t v : adjlist_[u])
					{
						for (id_t w : adjlist_[v])
						{
							if (w == id)
							{
								trianglecount++;
							}
						}
					}
				}
				return trianglecount;
			}
			// English note: original Chinese comment removed for anonymous release.

			void bfs(id_t s, std::vector<std::vector<id_t>> &adj, std::unordered_map<id_t, size_t> &dist, std::unordered_map<id_t, std::vector<id_t>> &pred, std::unordered_map<id_t, size_t> &sigma)
			{
				int n = adj.size();
				std::vector<bool> visited(n, false); // English note: original Chinese comment removed for anonymous release.

				dist[s] = 0;
				sigma[s] = 1;

				std::queue<id_t> q;
				q.push(s);
				visited[s] = true; // English note: original Chinese comment removed for anonymous release.

				while (!q.empty())
				{
					id_t v = q.front();
					q.pop();

					for (id_t u : adj[v])
					{
						// English note: original Chinese comment removed for anonymous release.
						if (!visited[u])
						{					   // English note: original Chinese comment removed for anonymous release.
							visited[u] = true; // English note: original Chinese comment removed for anonymous release.
							dist[u] = dist[v] + 1;
							q.push(u);
						}
						if (dist[u] == dist[v] + 1)
						{						  // English note: original Chinese comment removed for anonymous release.
							sigma[u] += sigma[v]; // English note: original Chinese comment removed for anonymous release.
							pred[u].push_back(v); // English note: original Chinese comment removed for anonymous release.
						}
					}
				}
			}

			// English note: original Chinese comment removed for anonymous release.
			/* English note: original Chinese comment removed for anonymous release. */

			// English note: original Chinese comment removed for anonymous release.
			std::vector<double> brandes_betweenness_centrality()
			{

				size_t n = adjlist_.size();
				std::vector<double> betweenness(n, 0.0);
				int count = 0;

				// English note: original Chinese comment removed for anonymous release.
				omp_set_num_threads(24);

// English note: original Chinese comment removed for anonymous release.
#pragma omp parallel for schedule(dynamic)
				for (id_t s = 0; s < n; ++s)
				{

					// std::cout << "cccNow:s=" << s << std::endl;
					/*
					if (s % 10 == 0)
					{
						std::cout << "Now:---" << s << std::endl;
					}*/
					Timer t;
					int time_bfs{0};
					int time_operate{0};
					std::unordered_map<id_t, size_t> dist, sigma;
					std::unordered_map<id_t, std::vector<id_t>> pred;

					// English note: original Chinese comment removed for anonymous release.
					bfs(s, adjlist_, dist, pred, sigma);

					// English note: original Chinese comment removed for anonymous release.
					std::vector<double> delta(n, 0.0);

					// English note: original Chinese comment removed for anonymous release.
					std::vector<id_t> nodes(n);
					for (id_t i = 0; i < n; ++i)
					{
						nodes[i] = i;
					}

					// English note: original Chinese comment removed for anonymous release.
					std::sort(nodes.begin(), nodes.end(), [&dist](id_t a, id_t b)
							  {
								  return dist[a] > dist[b]; // English note: original Chinese comment removed for anonymous release.
							  });

					for (id_t w : nodes)
					{
						if (w != s)
						{
							for (id_t v : pred[w])
							{
								delta[v] += (double(sigma[v]) / sigma[w]) * (1 + delta[w]);
							}
						}
					}

// English note: original Chinese comment removed for anonymous release.
#pragma omp critical // English note: original Chinese comment removed for anonymous release.
					{
						for (size_t w = 0; w < n; ++w)
						{
							if (w != s)
							{
								betweenness[w] += delta[w];
							}
						}

						count++;
						// cout<<"bfstime"<<time_bfs<<endl;
						// cout<<"operate"<<time_operate<<endl;

						if (count % 100 == 0)
						{
							cout << "Now:---" << count << endl;
						}
					}
				}

				// English note: original Chinese comment removed for anonymous release.
				for (int i = 0; i < n; ++i)
				{
					if (n > 2)
					{ // English note: original Chinese comment removed for anonymous release.
						betweenness[i] /= (double)((n - 1) * (n - 2));
					}
				}
				return betweenness;
			}

			// English note: original Chinese comment removed for anonymous release.
			float euclidean_distance(const std::vector<float> p1,const std::vector<float> p2)
			{
				
				
				float sum = 0;
				//sum += std::pow((p1[i] - p2[i]), 2);
				
				sum=euclidean(p1.data(), p2.data(), p1.size());
				//return sum;
				return sqrt(sum);
			}

			// English note: original Chinese comment removed for anonymous release.
			std::vector<id_t> region_query(id_t index)
			{
				std::vector<id_t> neighbors;
				for (int i = 0; i < pointcoordinates_.size(); ++i)
				{
					// English note: original Chinese comment removed for anonymous release.

					if (euclidean_distance(pointcoordinates_[index], pointcoordinates_[i]) <= eps_ * eps_)
					{
						neighbors.push_back(i);
					}
				}
				return neighbors;
			}
			// English note: original Chinese comment removed for anonymous release.
			void dbscan()
			{
				id_t cluster_id = 0;
				id_t n = pointcoordinates_.size();
				labels_.assign(n, 0); // English note: original Chinese comment removed for anonymous release.

				for (id_t i = 0; i < n; ++i)
				{
					// English note: original Chinese comment removed for anonymous release.
					if (labels_[i] != 0)
					{
						continue;
					}

					cout << "i=" << i << endl;

					// English note: original Chinese comment removed for anonymous release.
					std::vector<id_t> neighbors = region_query(i);
					// cout << "neighbors.size()=" << neighbors.size() << endl;

					// English note: original Chinese comment removed for anonymous release.
					if (neighbors.size() < minPts_)
					{
						labels_[i] = -1; // English note: original Chinese comment removed for anonymous release.
					}
					else
					{
						cluster_id++;			 // English note: original Chinese comment removed for anonymous release.
						labels_[i] = cluster_id; // English note: original Chinese comment removed for anonymous release.
												 // English note: original Chinese comment removed for anonymous release.

						// English note: original Chinese comment removed for anonymous release.
						std::unordered_set<id_t> visited_neighbors(neighbors.begin(), neighbors.end());

						// English note: original Chinese comment removed for anonymous release.
						for (size_t j = 0; j < neighbors.size(); ++j)
						{
							int neighbor_index = neighbors[j];

							// English note: original Chinese comment removed for anonymous release.
							if (labels_[neighbor_index] == 0)
							{
								labels_[neighbor_index] = cluster_id; // English note: original Chinese comment removed for anonymous release.

								// English note: original Chinese comment removed for anonymous release.
								std::vector<id_t> neighbor_neighbors = region_query(neighbor_index);
								// std::cout << "neighbor_neighbors.size()=" << neighbor_neighbors.size() << std::endl;

								// English note: original Chinese comment removed for anonymous release.
								if (neighbor_neighbors.size() >= minPts_)
								{

									for (id_t nn : neighbor_neighbors)
									{
										// English note: original Chinese comment removed for anonymous release.
										if (labels_[nn] == 0 && visited_neighbors.find(nn) == visited_neighbors.end())
										{
											visited_neighbors.insert(nn); // English note: original Chinese comment removed for anonymous release.
											neighbors.push_back(nn);	  // English note: original Chinese comment removed for anonymous release.
										}
									}
								}
							}
							cout << "neighbors.size()=" << neighbors.size() << endl;
						}
					}
				}
			}
			// English note: original Chinese comment removed for anonymous release.
			// English note: original Chinese comment removed for anonymous release.
			std::vector<int> bfsforAP(const std::vector<std::vector<int>> &adjList, int startNode)
			{
				int nodeCount = adjList.size();
				std::vector<int> distances(nodeCount, -1); // English note: original Chinese comment removed for anonymous release.
				distances[startNode] = 0;

				std::queue<int> q;
				q.push(startNode);

				while (!q.empty())
				{
					int node = q.front();
					q.pop();

					for (int neighbor : adjList[node])
					{
						if (distances[neighbor] == -1)
						{ // English note: original Chinese comment removed for anonymous release.
							distances[neighbor] = distances[node] + 1;
							q.push(neighbor);
						}
					}
				}

				return distances;
			}
			// English note: original Chinese comment removed for anonymous release.
		public:
			// English note: original Chinese comment removed for anonymous release.
			void point_data_in(const std::string &filename)
			{
				std::ifstream file(filename, std::ios::binary);
				if (!file.is_open())
				{
					std::cerr << "Error opening file: " << filename << std::endl;
					throw;
				}

				int D; // English note: original Chinese comment removed for anonymous release.
				using T = float;
				file.read(reinterpret_cast<char *>(&D), sizeof(int));

				file.seekg(0, std::ios::end);
				size_t file_size = file.tellg();
				// English note: original Chinese comment removed for anonymous release.
				size_t N = (file_size) / ((D) * sizeof(T) + sizeof(int));

				file.seekg(0, std::ios::beg);

				// data.resize(sizeof(T) * D);
				// English note: original Chinese comment removed for anonymous release.
				// data.shrink_to_fit();
				std::vector<T> data(D);
				int sep;
				for (size_t n = 0; n < N; ++n)
				{
					// std::vector<T> data(D * sizeof(T));
					file.read(reinterpret_cast<char *>(&sep), sizeof(int));
					file.read(reinterpret_cast<char *>(data.data()), D * sizeof(T));

					pointcoordinates_.push_back(data);
				}
				// printf("%s: [%zu x %d] has loaded!\n", filename.data(), N, D);
				file.close();

				N_ = N;
				D_ = D;
				// return { N, D };
			}
			// English note: original Chinese comment removed for anonymous release.
			int adj_data_in(const std::string &filename, int i)
			{
				// std::vector < std::vector<id_t> > adj_list;
				std::ifstream in(filename, std::ios::binary);
				if (!in.is_open())
				{
					throw std::runtime_error("Cannot open file for reading");
				}
				if (i == 0)
				{
					// hcnng
					// English note: original Chinese comment removed for anonymous release.

					// English note: original Chinese comment removed for anonymous release.
					size_t temp1;
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
				}
				else if (i == 1)
				{

					// vamana
					// English note: original Chinese comment removed for anonymous release.
					size_t temp1;
					float temp2;
					id_t temp3;
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					// float
					in.read(reinterpret_cast<char *>(&temp2), sizeof(temp2));
					// id_t
					in.read(reinterpret_cast<char *>(&temp3), sizeof(temp3));
				}
				else if (i == 2)
				{
					// nsg
					size_t temp1;
					float temp2;
					int temp3;
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					// id_t
					in.read(reinterpret_cast<char *>(&temp3), sizeof(temp3));
				}
				else if (i == 3)
				{
					// hnsw
					size_t temp1;
					double temp2;
					int temp3;
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));
					in.read(reinterpret_cast<char *>(&temp1), sizeof(temp1));

					in.read(reinterpret_cast<char *>(&temp2), sizeof(temp2));
					in.read(reinterpret_cast<char *>(&temp2), sizeof(temp2));

					in.read(reinterpret_cast<char *>(&temp3), sizeof(temp3));
					in.read(reinterpret_cast<char *>(&temp3), sizeof(temp3));
					in.read(reinterpret_cast<char *>(&temp3), sizeof(temp3));
					// id_t
					// English note: original Chinese comment removed for anonymous release.
					std::vector<int> element_levels1; // English note: original Chinese comment removed for anonymous release.
					size_t size = N_ /* English note: original Chinese comment removed for anonymous release. */;  // English note: original Chinese comment removed for anonymous release.
					printf("Info: %d", size);
					element_levels1.resize(size);
					// English note: original Chinese comment removed for anonymous release.
					in.read(reinterpret_cast<char *>(element_levels1.data()), size * sizeof(int)); // English note: original Chinese comment removed for anonymous release.
					// English note: original Chinese comment removed for anonymous release.
					// English note: original Chinese comment removed for anonymous release.
					size_t num_count = 0; // English note: original Chinese comment removed for anonymous release.

					while (true)
					{
						//printf("num_count:%d\n", num_count);
						size_t n=0;
						// size_t num_count= 0;
						// English note: original Chinese comment removed for anonymous release.
						// English note: original Chinese comment removed for anonymous release.
						if(num_count>=1000000){
							break;
						}
						if (element_levels1[num_count] != 0)
						{
							// English note: original Chinese comment removed for anonymous release.
							for (int j = 0; j <= element_levels1[num_count]; j++)
							{
								if (!in.read(reinterpret_cast<char *>(&n), sizeof(n)))
								{
									printf("Info.");
									printf("Info: %d",num_count);
									calculateDegrees();
									in.close(); // English note: original Chinese comment removed for anonymous release.
									return 0; // English note: original Chinese comment removed for anonymous release.
								}
								if (j == 0) // English note: original Chinese comment removed for anonymous release.
								{
									
									std::vector<id_t> neighbors(n);										   // English note: original Chinese comment removed for anonymous release.
									in.read(reinterpret_cast<char *>(neighbors.data()), n * sizeof(id_t)); // English note: original Chinese comment removed for anonymous release.
									adjlist_.push_back(std::move(neighbors));
									continue;
								}

								std::vector<id_t> neighbors(n);										   // English note: original Chinese comment removed for anonymous release.
								in.read(reinterpret_cast<char *>(neighbors.data()), n * sizeof(id_t)); // English note: original Chinese comment removed for anonymous release.
							}
							// English note: original Chinese comment removed for anonymous release.
							num_count++; // English note: original Chinese comment removed for anonymous release.
							continue;
						}
						// English note: original Chinese comment removed for anonymous release.

						// English note: original Chinese comment removed for anonymous release.
						if (!in.read(reinterpret_cast<char *>(&n), sizeof(n)))
						{
							break; // English note: original Chinese comment removed for anonymous release.
						}
						std::vector<id_t> neighbors(n);										   // English note: original Chinese comment removed for anonymous release.
						in.read(reinterpret_cast<char *>(neighbors.data()), n * sizeof(id_t)); // English note: original Chinese comment removed for anonymous release.
						num_count++;
						adjlist_.push_back(std::move(neighbors)); // English note: original Chinese comment removed for anonymous release.
					}
					in.close(); // English note: original Chinese comment removed for anonymous release.
					printf("num_count: %d\n",num_count);
					printf("adjlist_.size(): %d\n",adjlist_.size());
					calculateDegrees();
					//printf("Info.", adjlist_.size());
					// printf(adjlist_.size());
					return 0;
				}

				// English note: original Chinese comment removed for anonymous release.
				//printf("read done111/n");
				while (true)
				{
					size_t n;
					// English note: original Chinese comment removed for anonymous release.

					if (!in.read(reinterpret_cast<char *>(&n), sizeof(n)))
					{
						break; // English note: original Chinese comment removed for anonymous release.
					}

					std::vector<id_t> neighbors(n); // English note: original Chinese comment removed for anonymous release.

					in.read(reinterpret_cast<char *>(neighbors.data()), n * sizeof(id_t)); // English note: original Chinese comment removed for anonymous release.

					adjlist_.push_back(std::move(neighbors)); // English note: original Chinese comment removed for anonymous release.
				}
				//printf("read done123/n");
				in.close(); // English note: original Chinese comment removed for anonymous release.
				calculateDegrees();
				return 0;
				// cout<<
				// English note: original Chinese comment removed for anonymous release.
			}

			// English note: original Chinese comment removed for anonymous release.
			double calculateAssortativity()
			{
				// std::vector<int> out_degrees, in_degrees;

				// calculateDegrees();

				// English note: original Chinese comment removed for anonymous release.
				std::unordered_map<int, double> ax, by;
				std::unordered_map<std::pair<int, int>, double, pair_hash> exy;

				// English note: original Chinese comment removed for anonymous release.
				calculateDistributions(ax, by, exy);

				// English note: original Chinese comment removed for anonymous release.
				double assortativity_coefficient = CalculateAssortativity_core(ax, by, exy);
				return assortativity_coefficient;
			}
			// English note: original Chinese comment removed for anonymous release.
			double Cluster_coefficient()
			{
				int N = adjlist_.size()/20;
				double total_cc = 0.0;

				// calculateDegrees();

				for (id_t i = 0; i < N; i++)
				{
					int degree = in_degrees_[i]; // English note: original Chinese comment removed for anonymous release.
					if(i%1000==0)
					{
						printf("i=%d\n",i);
					}
					// English note: original Chinese comment removed for anonymous release.
					if (degree < 2)
					{
						continue;
					}

					size_t triangles = countTriangles(i, degree);

					// std::cout << "Info." << i << "Info." << triangles << std::endl;
					// English note: original Chinese comment removed for anonymous release.
					// English note: original Chinese comment removed for anonymous release.
					// English note: original Chinese comment removed for anonymous release.
					double u_cc = (1.0 * triangles) / (in_degrees_[i] * out_degrees_[i]); // English note: original Chinese comment removed for anonymous release.

					// std::cout << "u_cc:" << u_cc << std::endl;

					total_cc = total_cc + u_cc;
				}
				return total_cc / N;
			}
			// English note: original Chinese comment removed for anonymous release.
			void pathlengthdistribution()
			{
				int N = adjlist_.size();
				size_t dim = pointcoordinates_[0].size();
				for (int u = 0; u < N; u++)
				{
					for (int v : adjlist_[u])
					{
						float distance = 0;
						distance = euclidean_distance(pointcoordinates_[u], pointcoordinates_[v]);
						/*
						for (int i = 0; i < dim; i++)
						{

							distance += (pointcoordinates_[u][i] - pointcoordinates_[v][i]) * (pointcoordinates_[u][i] - pointcoordinates_[v][i]);
						}*/
						//distance = std::sqrt(distance);

						// English note: original Chinese comment removed for anonymous release.
						distances_.emplace_back(u, v, distance);
					}
				}
			}

			//------------------------
			void degreedistribution()
			{
				calculateDegrees();
				// return std::make_pair(in_degrees_, out_degrees_);
			}

			void between_centrality()
			{

				between_centrality_ = brandes_betweenness_centrality();
				int a = 1;
				// English note: original Chinese comment removed for anonymous release.
				/*
				cout << "Betweenness Centrality:" << endl;
				for (size_t i = 0; i < bc.size(); ++i) {
					cout << "Node " << i << ": " << bc[i] << endl;
				}*/
			}
			void calculatePageRank()
			{
				id_t n = N_;
				pagerank_.resize(n); // English note: original Chinese comment removed for anonymous release.
				// English note: original Chinese comment removed for anonymous release.
				std::transform(pagerank_.begin(), pagerank_.end(), pagerank_.begin(),
							   [n](double)
							   { return 1.0 / n; });

				// English note: original Chinese comment removed for anonymous release.
				std::vector<double> newPageRank(n, 0.0); // English note: original Chinese comment removed for anonymous release.

				for (int iteration = 0; iteration < maxIterations_; ++iteration)
				{
					// English note: original Chinese comment removed for anonymous release.
					for (id_t i = 0; i < n; ++i)
					{
						newPageRank[i] = (1.0 - dampingFactor_) / n; // English note: original Chinese comment removed for anonymous release.
						if (!adjlist_[i].empty())
						{
							for (id_t j : adjlist_[i])
							{
								newPageRank[i] += dampingFactor_ * (pagerank_[j] / adjlist_[j].size()); // English note: original Chinese comment removed for anonymous release.
							}
						}
					}

					// English note: original Chinese comment removed for anonymous release.
					double diff = 0.0;
					for (int i = 0; i < n; ++i)
					{
						diff += abs(newPageRank[i] - pagerank_[i]);
					}
					if (diff < tolerance_)
					{
						break;
					}

					// English note: original Chinese comment removed for anonymous release.
					pagerank_ = newPageRank;
				}
				int i = 1;
				/*
				cout << "PageRank values:" << endl;
				for (size_t i = 0; i < pageRank.size(); ++i) {
					cout << "Node " << i << ": " << pageRank[i] << endl;
				}*/
			}
			// pagerank

			void vector_relevant()
			{
				size_t dimensions = pointcoordinates_[0].size();
				size_t num_samples = pointcoordinates_.size();

				// English note: original Chinese comment removed for anonymous release.
				std::cout << "Number of dimensions: " << dimensions << std::endl;
				std::cout << "Number of samples: " << num_samples << std::endl;

				// English note: original Chinese comment removed for anonymous release.
				means_.resize(dimensions);
				variances_.resize(dimensions);
				skewnesses_.resize(dimensions);
				kurtoses_.resize(dimensions);

				// English note: original Chinese comment removed for anonymous release.
				std::fill(means_.begin(), means_.end(), 0);
				std::fill(variances_.begin(), variances_.end(), 0);
				std::fill(skewnesses_.begin(), skewnesses_.end(), 0);
				std::fill(kurtoses_.begin(), kurtoses_.end(), 0);

				// English note: original Chinese comment removed for anonymous release.
				for (const auto &vec : pointcoordinates_)
				{
					for (size_t i = 0; i < dimensions; ++i)
					{
						means_[i] += vec[i];
					}
				}
				for (size_t i = 0; i < dimensions; ++i)
				{
					means_[i] /= num_samples;
				}

				// English note: original Chinese comment removed for anonymous release.
				for (const auto &vec : pointcoordinates_)
				{
					for (size_t i = 0; i < dimensions; ++i)
					{
						double diff = vec[i] - means_[i];
						variances_[i] += diff * diff;
						skewnesses_[i] += diff * diff * diff;
						kurtoses_[i] += diff * diff * diff * diff;
					}
				}

				for (size_t i = 0; i < dimensions; ++i)
				{
					variances_[i] /= num_samples;
					skewnesses_[i] /= num_samples;
					kurtoses_[i] /= num_samples;
				}

				// English note: original Chinese comment removed for anonymous release.
				for (size_t i = 0; i < dimensions; ++i)
				{
					if (variances_[i] != 0)
					{ // English note: original Chinese comment removed for anonymous release.
						skewnesses_[i] /= std::pow(std::sqrt(variances_[i]), 3);
						kurtoses_[i] = (kurtoses_[i] / std::pow(variances_[i], 2)) - 3; // English note: original Chinese comment removed for anonymous release.
					}
				}

				// English note: original Chinese comment removed for anonymous release.
				/*
				for (size_t i = 0; i < dimensions; ++i) {
					std::cout << "Dimension " << i + 1 << ":\n";
					std::cout << "  Mean: " << means_[i] << "\n";
					std::cout << "  Variance: " << variances_[i] << "\n";
					std::cout << "  Skewness: " << skewnesses_[i] << "\n";
					std::cout << "  Kurtosis: " << kurtoses_[i] << "\n";
				}*/
			}
			double silhouette_coefficient()
			{
				id_t n = pointcoordinates_.size();
				double total_silhouette = 0.0;
				// English note: original Chinese comment removed for anonymous release.

				cout << "Info." << endl;
				dbscan();
				cout << "Info." << endl;
				size_t num_clusters = *max_element(labels_.begin(), labels_.end()) + 1; // English note: original Chinese comment removed for anonymous release.

				for (id_t i = 0; i < n; ++i)
				{
					id_t label_i = labels_[i];
					if (label_i == -1)
						continue; // English note: original Chinese comment removed for anonymous release.

					double a = 0.0;								   // English note: original Chinese comment removed for anonymous release.
					double b = std::numeric_limits<double>::max(); // English note: original Chinese comment removed for anonymous release.
					int count_a = 0;

					// English note: original Chinese comment removed for anonymous release.
					for (id_t j = 0; j < n; ++j)
					{
						if (labels_[j] == label_i && i != j)
						{
							a += euclidean_distance(pointcoordinates_[i], pointcoordinates_[j]);
							count_a++;
						}
					}
					if (count_a > 0)
					{
						a /= count_a; // English note: original Chinese comment removed for anonymous release.
					}

					// English note: original Chinese comment removed for anonymous release.
					for (int k = 0; k < n; ++k)
					{
						if (labels_[k] != label_i && labels_[k] != -1)
						{ // English note: original Chinese comment removed for anonymous release.
							double distance = euclidean_distance(pointcoordinates_[i], pointcoordinates_[k]);
							double sum_b = 0.0;
							int count_b = 0;

							// English note: original Chinese comment removed for anonymous release.
							for (id_t l = 0; l < n; ++l)
							{
								if (labels_[l] == labels_[k])
								{
									sum_b += euclidean_distance(pointcoordinates_[i], pointcoordinates_[l]);
									count_b++;
								}
							}

							if (count_b > 0)
							{
								sum_b /= count_b; // English note: original Chinese comment removed for anonymous release.
							}

							if (sum_b < b)
							{
								b = sum_b; // English note: original Chinese comment removed for anonymous release.
							}
						}
					}

					// English note: original Chinese comment removed for anonymous release.
					if (count_a > 0 && b != std::numeric_limits<double>::max())
					{
						double s = (b - a) / std::max(a, b);
						total_silhouette += s;
					}
				}

				return total_silhouette / n; // English note: original Chinese comment removed for anonymous release.
			}

			double calinski_harabasz_index()
			{
				id_t n = pointcoordinates_.size();
				size_t k = *max_element(labels_.begin(), labels_.end()) + 1; // English note: original Chinese comment removed for anonymous release.
				if (k <= 1)
				{
					return 0.0;
				} // English note: original Chinese comment removed for anonymous release.

				// English note: original Chinese comment removed for anonymous release.
				std::vector<float> global_mean(pointcoordinates_[0].size(), 0.0);
				for (const auto &point : pointcoordinates_)
				{
					for (size_t j = 0; j < point.size(); ++j)
					{
						global_mean[j] += point[j];
					}
				}
				for (auto &val : global_mean)
				{
					val /= n; // English note: original Chinese comment removed for anonymous release.
				}

				double B_k = 0.0; // English note: original Chinese comment removed for anonymous release.
				double W_k = 0.0; // English note: original Chinese comment removed for anonymous release.

				// English note: original Chinese comment removed for anonymous release.
				std::unordered_map<id_t, std::vector<float>> cluster_means;
				std::unordered_map<id_t, id_t> cluster_counts;

				for (id_t i = 0; i < n; ++i)
				{
					id_t label = labels_[i];
					cluster_counts[label]++;
					if (cluster_means.find(label) == cluster_means.end())
					{
						cluster_means[label] = std::vector<float>(pointcoordinates_[i].size(), 0.0);
					}
					for (size_t j = 0; j < pointcoordinates_[i].size(); ++j)
					{
						cluster_means[label][j] += pointcoordinates_[i][j];
					}
				}

				for (auto &pair : cluster_means)
				{
					id_t label = pair.first;
					std::vector<float> &mean = pair.second;
					for (size_t j = 0; j < mean.size(); ++j)
					{
						mean[j] /= cluster_counts[label]; // English note: original Chinese comment removed for anonymous release.
					}
				}

				// English note: original Chinese comment removed for anonymous release.
				for (id_t i = 0; i < n; ++i)
				{
					id_t label = labels_[i];
					double dist_to_global = euclidean_distance(pointcoordinates_[i], global_mean);
					B_k += dist_to_global * dist_to_global; // English note: original Chinese comment removed for anonymous release.

					double dist_to_mean = euclidean_distance(pointcoordinates_[i], cluster_means[label]);
					W_k += dist_to_mean; // English note: original Chinese comment removed for anonymous release.
				}

				return (B_k / (k - 1)) / (W_k / (n - k)); // English note: original Chinese comment removed for anonymous release.
			}

			double davies_bouldin_index()
			{
				id_t n = pointcoordinates_.size();
				size_t k = *max_element(labels_.begin(), labels_.end()) + 1; // English note: original Chinese comment removed for anonymous release.
				if (k <= 1)
				{
					return std::numeric_limits<double>::max();
				} // English note: original Chinese comment removed for anonymous release.

				std::vector<double> S(k, 0.0);										// English note: original Chinese comment removed for anonymous release.
				std::vector<std::vector<double>> D(k, std::vector<double>(k, 0.0)); // English note: original Chinese comment removed for anonymous release.

				// English note: original Chinese comment removed for anonymous release.
				std::unordered_map<id_t, std::vector<double>> cluster_means;
				std::unordered_map<id_t, id_t> cluster_counts;

				for (id_t i = 0; i < n; ++i)
				{
					id_t label = labels_[i];
					cluster_counts[label]++;

					// English note: original Chinese comment removed for anonymous release.
					if (cluster_means.find(label) == cluster_means.end())
					{
						cluster_means[label] = std::vector<double>(pointcoordinates_[i].size(), 0.0);
					}

					// English note: original Chinese comment removed for anonymous release.
					for (size_t j = 0; j < pointcoordinates_[i].size(); ++j)
					{
						cluster_means[label][j] += pointcoordinates_[i][j]; // English note: original Chinese comment removed for anonymous release.
					}
				}

				for (auto &pair : cluster_means)
				{
					id_t label = pair.first;
					std::vector<double> &mean = pair.second;
					for (size_t j = 0; j < mean.size(); ++j)
					{
						mean[j] /= cluster_counts[label]; // English note: original Chinese comment removed for anonymous release.
					}
				}

				for (id_t i = 0; i < n; ++i)
				{
					id_t label = labels_[i];
					double dist_to_mean = 0.0;
					for (size_t j = 0; j < pointcoordinates_[i].size(); ++j)
					{
						dist_to_mean += euclidean_distance(pointcoordinates_[i], cluster_means[label]);
					}
					S[label] += dist_to_mean; // English note: original Chinese comment removed for anonymous release.
				}

				for (id_t i = 0; i < k; ++i)
				{
					S[i] /= cluster_counts[i]; // English note: original Chinese comment removed for anonymous release.
				}

				// English note: original Chinese comment removed for anonymous release.
				for (id_t i = 0; i < k; ++i)
				{
					for (id_t j = i + 1; j < k; ++j)
					{
						D[i][j] = euclidean_distance(cluster_means[i], cluster_means[j]);
						D[j][i] = D[i][j]; // English note: original Chinese comment removed for anonymous release.
					}
				}

				// English note: original Chinese comment removed for anonymous release.
				double DB = 0.0;
				for (id_t i = 0; i < k; ++i)
				{
					double max_ratio = 0.0;
					for (id_t j = 0; j < k; ++j)
					{
						if (i != j)
						{
							double ratio = (S[i] + S[j]) / D[i][j];
							max_ratio = std::max(max_ratio, ratio);
						}
					}
					DB += max_ratio;
				}

				return DB / k; // English note: original Chinese comment removed for anonymous release.
			}

			// English note: original Chinese comment removed for anonymous release.
			double calculateAveragePathLength(int sampleCount)
			{
				size_t totalDistance = 0;
				size_t pathCount = 0;
				// English note: original Chinese comment removed for anonymous release.
				for (int i = 0; i < sampleCount; ++i)
				{
					if (i % (sampleCount / 10) == 0)
					{
						cout << "i:" << i << endl;
					}
					std::vector<int> distances = bfsforAP(adjlist_, i);
					for (int d : distances)
					{
						if (d > 0)
						{ // English note: original Chinese comment removed for anonymous release.
							totalDistance += d;
							pathCount++;
						}
					}
				}
				// cout << "Info." << (static_cast<double>(totalDistance) / pathCount) << endl;
				averagePathLength_ = (static_cast<double>(totalDistance) / pathCount);

				return pathCount > 0 ? static_cast<double>(totalDistance) / pathCount : 0;
			}

			// English note: original Chinese comment removed for anonymous release.
			int calculateDiameter(int sample)
			{
				int diameter = 0;
				// English note: original Chinese comment removed for anonymous release.
				for (int i = 0; i < sample; ++i)
				{
					std::vector<int> distances = bfsforAP(adjlist_, i);
					for (int d : distances)
					{
						if (d > diameter)
						{
							diameter = d;
						}
					}
					if (i % (sample / 10) == 0)
					{
						printf("i:%d\n", i);
					}
				}
				diameter_ = diameter;
				// printf("Info: %d\n", diameter);
				return diameter;
			}

			// English note: original Chinese comment removed for anonymous release.
			void calculateEffectiveSize()
			{
				int nodeCount = adjlist_.size();

				effectivesize_.resize(nodeCount, 0.0);

				for (int u = 0; u < nodeCount; ++u)
				{
					float N = static_cast<float>(adjlist_[u].size()); // English note: original Chinese comment removed for anonymous release.
					if (N == 0.0)
					{ // English note: original Chinese comment removed for anonymous release.
						effectivesize_[u] = 0.0;
						continue;
					}

					float T = 0.0; // English note: original Chinese comment removed for anonymous release.
					// English note: original Chinese comment removed for anonymous release.
					for (int neighbor1 : adjlist_[u])
					{
						for (int neighbor2 : adjlist_[u])
						{
							if (neighbor1 != neighbor2 &&
								std::find(adjlist_[neighbor1].begin(), adjlist_[neighbor1].end(), neighbor2) != adjlist_[neighbor1].end())
							{
								T += 1.0; // English note: original Chinese comment removed for anonymous release.
							}
						}
					}

					effectivesize_[u] = N - (2.0 * T) / N; // English note: original Chinese comment removed for anonymous release.
				}
			}

			// English note: original Chinese comment removed for anonymous release.
			void calculateWFClosenessCentrality(int sampleNum)
			{
				int nodeCount = sampleNum;
				// std::vector<double> wfCentrality(nodeCount, 0.0);
				wfCentrality_.resize(nodeCount, 0.0);

				for (int u = 0; u < nodeCount; ++u)
				{
					/*
					if (u % 10 == 0)
					{
						printf("u:%d\n", u);
					}*/

					std::vector<int> distances = bfsforAP(adjlist_, u);
					double reachableCount = 0.0; // English note: original Chinese comment removed for anonymous release.
					double sumDistances = 0.0;

					for (int v = 0; v < nodeCount; ++v)
					{
						if (u != v && distances[v] > 0)
						{						   // English note: original Chinese comment removed for anonymous release.
							reachableCount += 1.0; // English note: original Chinese comment removed for anonymous release.
							sumDistances += distances[v];
						}
					}

					if (reachableCount > 0.0)
					{ // English note: original Chinese comment removed for anonymous release.
						wfCentrality_[u] = ((reachableCount - 1.0) / (nodeCount - 1.0)) *
										   ((reachableCount - 1.0) / sumDistances); // English note: original Chinese comment removed for anonymous release.
					}
				}
			}
		};
	}
}