// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>
#include <unordered_map>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include "filter_utils.h"
#include "utils.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

#define PARTSIZE 10000000
#define ALIGNMENT 512

// custom types (for readability)
typedef tsl::robin_set<std::string> label_set;
typedef std::string path;

namespace po = boost::program_options;

template <class T> T div_round_up(const T numerator, const T denominator)
{
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

//V1
using pairIF = std::pair<size_t, float>;
struct cmpmaxstruct
{
    bool operator()(const pairIF &l, const pairIF &r)
    {
        return l.second < r.second;
    };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

//Version2
using tupleIFV = std::tuple<size_t, float, std::vector<float>>;
struct cmpmaxstructV2
{
    bool operator()(const tupleIFV &l, const tupleIFV &r)
    {
        return std::get<1>(l) < std::get<1>(r); // Compare distances (second element)
        //By defining the comparator such that it returns true when the distance of l is less than the distance of r, we ensure that the element with the largest distance has the highest priority.
    };
};
//priority queue definition
using maxPQTuple = std::priority_queue<tupleIFV, std::vector<tupleIFV>, cmpmaxstructV2>;

//Version3
using pairVF = std::pair<std::vector<float>, float>;
struct cmpmaxstructV3
{
    bool operator()(const pairVF &l, const pairVF &r)
    {
        return l.second < r.second;
    };
};
using maxPQVF = std::priority_queue<pairVF, std::vector<pairVF>, cmpmaxstructV3>;

//Version4
template <typename T>
using pairVT = std::pair<std::vector<T>, float>;

template <typename T>
struct cmpmaxstructV4
{
    bool operator()(const pairVT<T> &l, const pairVT<T> &r)
    {
        return l.second < r.second;
    };
};

template <typename T>
using maxPQVT = std::priority_queue<pairVT<T>, std::vector<pairVT<T>>, cmpmaxstructV4<T>>;


template <class T> T *aligned_malloc(const size_t n, const size_t alignment)
{
#ifdef _WINDOWS
    return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
    return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

//MARK: updated the first of the pair
template <typename T>
inline bool custom_dist(const std::pair<std::vector<T>, float> &a, const std::pair<std::vector<T>, float> &b)
{
    return a.second < b.second;
}

void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const uint64_t dim)
{
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t d = 0; d < num_points; ++d)
    // cblas_sdot function from the BLAS library, which calculates the dot product of two vectors.
//A value of 1 means the elements are contiguous in memory and no elements are skipped.
        points_l2sq[d] = cblas_sdot((int64_t)dim, matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1,
                                    matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1);
}

void distsq_to_points(const size_t dim,
                      float *dist_matrix, // Col Major, cols are queries, rows are points
                      size_t npoints, const float *const points,
                      const float *const points_l2sq, // points in Col major
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq, // queries in Col major
                      float *ones_vec = NULL)          // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];//The size of ones_vec is set to the maximum of nqueries and npoints to ensure it can be used for both points and queries.
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, points_l2sq, npoints,
                ones_vec, nqueries, (float)1.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, ones_vec, npoints,
                queries_l2sq, nqueries, (float)1.0, dist_matrix, npoints);
    if (ones_vec_alloc)
        delete[] ones_vec;
}

void inner_prod_to_points(const size_t dim,
                          float *dist_matrix, // Col Major, cols are queries, rows are points
                          size_t npoints, const float *const points, size_t nqueries, const float *const queries,
                          float *ones_vec = NULL) // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-1.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);

    if (ones_vec_alloc)
        delete[] ones_vec;
}

template <typename T>
void exact_knn(const size_t dim, const size_t k,
               //size_t *const closest_points,     // k * num_queries preallocated, col
                                                 // major, queries columns
               float *const dist_closest_points, // k * num_queries
                                                 // preallocated, Dist to
                                                 // corresponding closes_points
               std::vector<std::vector<std::vector<T>>> &vec_closest_points, //top k nodes' value vectors for each query -- so it is 3D
               size_t npoints,
               float *points_in, // points in Col major
               size_t nqueries, float *queries_in,
               diskann::Metric metric = diskann::Metric::L2) // queries in Col major
{
    ///MARK: removing redundant points here? n*n -- too much overhead and error prone

    float *points_l2sq = new float[npoints];
    float *queries_l2sq = new float[nqueries];
    //computes the squared L2 norm (or squared Euclidean norm) of each point in a given matrix of points. 
    compute_l2sq(points_l2sq, points_in, npoints, dim);
    compute_l2sq(queries_l2sq, queries_in, nqueries, dim);

    float *points = points_in;
    float *queries = queries_in;

///TODO: modify the codes for cosine distance
//     if (metric == diskann::Metric::COSINE)
//     { // we convert cosine distance as
//       // normalized L2 distnace
//         points = new float[npoints * dim];
//         queries = new float[nqueries * dim];
// #pragma omp parallel for schedule(static, 4096)
//         for (int64_t i = 0; i < (int64_t)npoints; i++)
//         {
//             float norm = std::sqrt(points_l2sq[i]);
//             if (norm == 0)
//             {
//                 norm = std::numeric_limits<float>::epsilon();
//             }
//             for (uint32_t j = 0; j < dim; j++)
//             {
//                 points[i * dim + j] = points_in[i * dim + j] / norm;
//             }
//         }

// #pragma omp parallel for schedule(static, 4096)
//         for (int64_t i = 0; i < (int64_t)nqueries; i++)
//         {
//             float norm = std::sqrt(queries_l2sq[i]);
//             if (norm == 0)
//             {
//                 norm = std::numeric_limits<float>::epsilon();
//             }
//             for (uint32_t j = 0; j < dim; j++)
//             {
//                 queries[i * dim + j] = queries_in[i * dim + j] / norm;
//             }
//         }
//         // recalculate norms after normalizing, they should all be one.
//         compute_l2sq(points_l2sq, points, npoints, dim);
//         compute_l2sq(queries_l2sq, queries, nqueries, dim);
//     }

    std::cout << "Going to compute " << k << " NNs for " << nqueries << " queries over " << npoints << " points in "
              << dim << " dimensions using";
    if (metric == diskann::Metric::INNER_PRODUCT)
        std::cout << " MIPS ";
    else if (metric == diskann::Metric::COSINE)
        std::cout << " Cosine ";
    else
        std::cout << " L2 ";
    std::cout << "distance fn. " << std::endl;

    size_t q_batch_size = (1 << 9);//512
    float *dist_matrix = new float[(size_t)q_batch_size * (size_t)npoints];

    for (size_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b)
    {
        int64_t q_b = b * q_batch_size;//query begin
        int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;//query end

        if (metric == diskann::Metric::L2 || metric == diskann::Metric::COSINE)
        {
            //compute the squared Euclidean distance and populate it in dist_matrix
            distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                             queries + (ptrdiff_t)q_b * (ptrdiff_t)dim, queries_l2sq + q_b);
        }
        else
        {
            inner_prod_to_points(dim, dist_matrix, npoints, points, q_e - q_b,
                                 queries + (ptrdiff_t)q_b * (ptrdiff_t)dim);
        }
        std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;


///TODO: what can access the data value via data or _data based on id; 
//if there is same dist, we further compare if the data value are also same. if same, remove one and only only one -- make sure k different nodes are returned
#pragma omp parallel for schedule(dynamic, 16)
        for (long long q = q_b; q < q_e; q++)
        {
            //maxPQIFCS point_dist;//priority queue --- 
            //make sure the frist k are unique nodes first
            maxPQVT<T> vec_dist;//pq of pair -- vector of float and dist
            std::vector<std::vector<T>> topKUniqueVectors;
            topKUniqueVectors.reserve(k);
            size_t begin = 0;

            for (size_t p = 0; p < npoints; p++){
                std::vector<T> vecVal(dim);
                for(size_t j = 0; j < dim; j++){
                    vecVal[j] = static_cast<T>(points[p * dim + j]);
                }

                bool duplicate = false;
                //skip it if this node is duplicate
                for (const auto &vec : topKUniqueVectors) {
                    if (vec == vecVal) {
                        duplicate = true;
                        break; // No need to continue if we found a duplicate
                    }
                }

                if (!duplicate){
                    vec_dist.emplace(vecVal, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
                    topKUniqueVectors.push_back(vecVal);
                }

                begin++;

                //pre fullfill the maxPQ to make it contain k
                if (topKUniqueVectors.size() == k){
                    break;
                }
            }

            for (size_t p = begin; p < npoints; p++)
            {   
                // add this point p to the PQ if it is smaller the top and not a duplicate
                if (vec_dist.top().second > dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]){

                    std::vector<T> vecVal(dim);
                    for(size_t j = 0; j < dim; j++){
                        vecVal[j] = static_cast<T>(points[p * dim + j]);
                    }

                    bool duplicate = false;
                    //skip it if this node is duplicate
                    for (const auto &vec : topKUniqueVectors) {
                        if (vec == vecVal) {
                            duplicate = true;
                            break; // No need to continue if we found a duplicate
                        }
                    }

                    if (!duplicate){
                        vec_dist.emplace(vecVal, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
                        topKUniqueVectors.push_back(vecVal);
                    }
                }
                    
                if (vec_dist.size() > k){
                    //remove the top vec from topKUniqueVectors
                    auto &vecToRemove = vec_dist.top().first;
                    auto it = std::remove_if(topKUniqueVectors.begin(), topKUniqueVectors.end(),
                             [&vecToRemove](const std::vector<T>& vec) {
                                 return vec == vecToRemove;
                             });
                    topKUniqueVectors.erase(it, topKUniqueVectors.end());
                    vec_dist.pop();
                }
            }


            for (ptrdiff_t l = 0; l < (ptrdiff_t)k; ++l)
            {
                //closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = std::get<0>(vec_dist.top()); //closest one is stored in the begining
                vec_closest_points[q][(ptrdiff_t)(k - 1 - l)] = vec_dist.top().first;
                dist_closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = vec_dist.top().second;
                vec_dist.pop();
            }
            assert(std::is_sorted(dist_closest_points + (ptrdiff_t)q * (ptrdiff_t)k,
                                  dist_closest_points + (ptrdiff_t)(q + 1) * (ptrdiff_t)k));
        }
        std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
    }

    delete[] dist_matrix;

    delete[] points_l2sq;
    delete[] queries_l2sq;

    if (metric == diskann::Metric::COSINE)
    {
        delete[] points;
        delete[] queries;
    }
}

template <typename T> inline int get_num_parts(const char *filename)
{
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(filename, std::ios::binary);
    std::cout << "Reading bin file " << filename << " ...\n";
    int npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&ndims_i32, sizeof(int));
    std::cout << "#pts = " << npts_i32 << ", #dims = " << ndims_i32 << std::endl;
    reader.close();
    uint32_t num_parts =
        (npts_i32 % PARTSIZE) == 0 ? npts_i32 / PARTSIZE : (uint32_t)std::floor(npts_i32 / PARTSIZE) + 1;
    std::cout << "Number of parts: " << num_parts << std::endl;
    return num_parts;
}

template <typename T>
inline void load_bin_as_float(const char *filename, float *&data, size_t &npts, size_t &ndims, int part_num)
{
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(filename, std::ios::binary);
    std::cout << "Reading bin file " << filename << " ...\n";
    int npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&ndims_i32, sizeof(int));
    uint64_t start_id = part_num * PARTSIZE;
    uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t)npts_i32);
    npts = end_id - start_id;
    ndims = (uint64_t)ndims_i32;
    std::cout << "#pts in part = " << npts << ", #dims = " << ndims << ", size = " << npts * ndims * sizeof(T) << "B"
              << std::endl;

    reader.seekg(start_id * ndims * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);
    T *data_T = new T[npts * ndims];
    reader.read((char *)data_T, sizeof(T) * npts * ndims);
    std::cout << "Finished reading part of the bin file." << std::endl;
    reader.close();
    data = aligned_malloc<float>(npts * ndims, ALIGNMENT);
#pragma omp parallel for schedule(dynamic, 32768)
    for (int64_t i = 0; i < (int64_t)npts; i++)
    {
        for (int64_t j = 0; j < (int64_t)ndims; j++)
        {
            float cur_val_float = (float)data_T[i * ndims + j];
            std::memcpy((char *)(data + i * ndims + j), (char *)&cur_val_float, sizeof(float));
        }
    }
    delete[] data_T;
    std::cout << "Finished converting part data to float." << std::endl;
}

template <typename T> inline void save_bin(const std::string filename, T *data, size_t npts, size_t ndims)
{
    std::ofstream writer;
    writer.exceptions(std::ios::failbit | std::ios::badbit);
    writer.open(filename, std::ios::binary | std::ios::out);
    std::cout << "Writing bin: " << filename << "\n";
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    std::cout << "bin: #pts = " << npts << ", #dims = " << ndims
              << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int) << "B" << std::endl;

    writer.write((char *)data, npts * ndims * sizeof(T));
    writer.close();
    std::cout << "Finished writing bin" << std::endl;
}
//npts is the queries while ndims is K
template <typename T>
inline void save_groundtruth_as_one_file(const std::string filename, const T *vec_closest_points, const float *distances, size_t npts,
                                         size_t ndims, size_t vdims)
{
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims, vdims_i32 = (int)vdims;
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    writer.write((char *)&vdims_i32, sizeof(int));
    ///MARK: issue is due to here ---- not sizeof float but of T
    std::cout << "Saving truthset in one file (npts, gt_dim, npts*gt_dim*vec_dim vectors of ground truth, "
                 "npts*gt_dim dist-matrix) with npts = "
              << npts << ", gt_dim = " << ndims << ", vec_dim = " << vdims << ", size = " <<  npts * ndims * vdims * sizeof(T) + npts * ndims * sizeof(float) + 3 * sizeof(int)
              << "B" << std::endl;

    writer.write((char *)vec_closest_points, npts * ndims * vdims * sizeof(T));
    writer.write((char *)distances, npts * ndims * sizeof(float));
    writer.close();
    std::cout << "Finished writing truthset" << std::endl;
}

template <typename T>
std::vector<std::vector<std::pair<std::vector<T>, float>>> processUnfilteredParts(const std::string &base_file,
                                                                            size_t &nqueries, size_t &npoints,
                                                                            size_t &dim, size_t &k, float *query_data,
                                                                            const diskann::Metric &metric,
                                                                            std::vector<uint32_t> &location_to_tag)
{
    float *base_data = nullptr;
    int num_parts = get_num_parts<T>(base_file.c_str());
    std::vector<std::vector<std::pair<std::vector<T>, float>>> res(nqueries);//1st D -- query; 2nd D: K * num_parts; 3D -- basic element -- pair
    for (int p = 0; p < num_parts; p++)
    {
        size_t start_id = p * PARTSIZE;
        //load data from p-th part; npoints is the num of points within this part
        load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p);

        //size_t *closest_points_part = new size_t[nqueries * k];
        float *dist_closest_points_part = new float[nqueries * k];

        auto part_k = k < npoints ? k : npoints;//get the smaller one
        //get the top K from this part -- and then sort among all parts
        //npoints is the num of points within this part
        std::vector<std::vector<std::vector<T>>> vec_closest_points(nqueries, std::vector<std::vector<T>>(k));//The innermost vector is initially not set to any specific size; it is just an empty std::vector<float>.
        exact_knn<T>(dim, part_k, dist_closest_points_part, vec_closest_points, npoints, base_data, nqueries, query_data,
                  metric);

        for (size_t i = 0; i < nqueries; i++)
        {
            for (size_t j = 0; j < part_k; j++)//for each points of the top K
            {   
                if (!location_to_tag.empty())
                    diskann::cout << "Error: location_to_tag should be empty." << std::endl;
                    // if (location_to_tag[closest_points_part[i * k + j] + start_id] == 0)
                    //     continue;
                //note: this is pushing back not assigning. that is why it works!
                res[i].push_back(std::make_pair(vec_closest_points[i][j],
                                                dist_closest_points_part[i * part_k + j]));
            }
        }

        //delete[] closest_points_part;
        delete[] dist_closest_points_part;

        diskann::aligned_free(base_data);
    }
    return res;
};

template <typename T>
int aux_main(const std::string &base_file, const std::string &query_file, const std::string &gt_file, size_t k,
             const diskann::Metric &metric, const std::string &tags_file = std::string(""))
{
    size_t npoints, nqueries, dim;

    float *query_data;
//std::cout << "Place 1" << std::endl;
    load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);
//std::cout << "Place 1.1" << std::endl;
    if (nqueries > PARTSIZE)
        std::cerr << "WARNING: #Queries provided (" << nqueries << ") is greater than " << PARTSIZE
                  << ". Computing GT only for the first " << PARTSIZE << " queries." << std::endl;

    // load tags
    const bool tags_enabled = tags_file.empty() ? false : true;
    std::vector<uint32_t> location_to_tag = diskann::loadTags(tags_file, base_file);

    //int *closest_points = new int[nqueries * k];
    float *dist_closest_points = new float[nqueries * k];
    T *vec_closest_points_flatten = new T[nqueries * k * dim];
    std::vector<std::vector<std::vector<T>>> vec_closest_points(nqueries);//queries, top K, vec of floats
//std::cout << "Place 2" << std::endl;
///MARK: need to update: 
    std::vector<std::vector<std::pair<std::vector<T>, float>>> results =
        processUnfilteredParts<T>(base_file, nqueries, npoints, dim, k, query_data, metric, location_to_tag);
//std::cout << "Place 3" << std::endl;
    for (size_t i = 0; i < nqueries; i++)
    {
        std::vector<std::pair<std::vector<T>, float>> &cur_res = results[i];//cur_res is the K * num_parts of pairs for this query i
        std::sort(cur_res.begin(), cur_res.end(), custom_dist<T>);//a.second < b.second;
        size_t j = 0;
        for (auto iter : cur_res)//each iter is a pair
        {
            if (j == k)
                break;
                     
            //if duplicate continue; else, execute the rest code
            std::vector<T> vecVal = iter.first;

            bool duplicate = false;
            //skip it if this node is duplicate
            for (const auto &vec : vec_closest_points[i]) {
                if (vec == vecVal) {
                    duplicate = true;
                    break; // No need to continue if we found a duplicate
                }
            }

            if (duplicate){
                continue;
            }

            vec_closest_points[i].push_back(vecVal);
            dist_closest_points[i * k + j] = iter.second;
            //flatten the vec data into a 1D vector so that we can easily write this continuous data into file
            for (size_t d = 0; d < dim; d++){
                vec_closest_points_flatten[(i * k * dim) + (j * dim) + d] = vecVal[d];
            }

            ++j;
        }
        if (j < k)
            std::cout << "WARNING: found less than k GT entries for query " << i << std::endl;
    }
//std::cout << "Place 4" << std::endl;
///MARK: need to update: 
    save_groundtruth_as_one_file<T>(gt_file, vec_closest_points_flatten, dist_closest_points, nqueries, k, dim);
    //delete[] closest_points;
    delete[] dist_closest_points;
    delete[] vec_closest_points_flatten;
    diskann::aligned_free(query_data);

    return 0;
}

// void load_truthset(const std::string &bin_file, float *&dists, size_t &npts, size_t &dim, std::vector<std::vector<std::vector<float>>> &gt_values)
// {
//     size_t read_blk_size = 64 * 1024 * 1024;
//     cached_ifstream reader(bin_file, read_blk_size);
//     diskann::cout << "Reading truthset file " << bin_file.c_str() << " ..." << std::endl;
//     size_t actual_file_size = reader.get_file_size();

//     int npts_i32, dim_i32, vec_dim_i32;
//     reader.read((char *)&npts_i32, sizeof(int));
//     reader.read((char *)&dim_i32, sizeof(int));
//     reader.read((char *)&vec_dim_i32, sizeof(int));
//     npts = (uint32_t)npts_i32;
//     dim = (uint32_t)dim_i32;
//     size_t vdims = (uint32_t)vec_dim_i32;

//     diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << ", #vec_dims = " << vdims << "... " << std::endl;

//     int truthset_type = -1; // 1 means truthset has values and distances, 2 means only ids, -1 is error

//     size_t expected_file_size_with_dists = npts * dim * vdims * sizeof(float) + npts * dim * sizeof(float) + 3 * sizeof(int);

//     if (actual_file_size == expected_file_size_with_dists)
//         truthset_type = 1;

//     size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

//     if (actual_file_size == expected_file_size_just_ids)
//         truthset_type = 2;

//     if (truthset_type == -1)
//     {
//         std::stringstream stream;
//         stream << "Error. File size mismatch. File should have bin format, with "
//                   "npts followed by ngt followed by npts*ngt ids and optionally "
//                   "followed by npts*ngt distance values; actual size: "
//                << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
//                << expected_file_size_just_ids;
//         diskann::cout << stream.str();
//         throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
//     }

//     //ids = new uint32_t[npts * dim];
//     //reader.read((char *)ids, npts * dim * sizeof(uint32_t));
//     float *gt_values_flatten = new float[npts * dim * vdims];
//     reader.read((char *)gt_values_flatten, npts * dim * vdims * sizeof(float));

//     //covert the flatten to 3D vector -- std::vector<std::vector<std::vector<float>>> gt_values(query_num);
//     for (size_t i = 0; i < npts; i++) {
//         gt_values[i].resize(dim); //  resizes the i-th vector to have dim elements, each of which is an empty vector of float.
//         for (size_t j = 0; j < dim; j++) {
//             gt_values[i][j].resize(vdims); // resizes the j-th vector within the i-th vector to have vdims elements, each initialized to 0.
//             for (size_t d = 0; d < vdims; d++) {
//                 gt_values[i][j][d] = gt_values_flatten[(i * dim * vdims) + (j * vdims) + d];//replace 0 with real float value
//             }
//         }
//     }

//     if (truthset_type == 1)
//     {
//         dists = new float[npts * dim];
//         reader.read((char *)dists, npts * dim * sizeof(float));
//     }

//     delete[] gt_values_flatten;
// }

// template <typename T>
// inline void load_truthset(const std::string &bin_file, float *&dists, size_t &npts, size_t &dim, std::vector<std::vector<std::vector<T>>> &gt_values)
// {
//     size_t read_blk_size = 64 * 1024 * 1024;
//     cached_ifstream reader(bin_file, read_blk_size);
//     diskann::cout << "Reading truthset file " << bin_file.c_str() << " ..." << std::endl;
//     size_t actual_file_size = reader.get_file_size();

//     int npts_i32, dim_i32, vec_dim_i32;
//     reader.read((char *)&npts_i32, sizeof(int));
//     reader.read((char *)&dim_i32, sizeof(int));
//     reader.read((char *)&vec_dim_i32, sizeof(int));
//     npts = (uint32_t)npts_i32;
//     dim = (uint32_t)dim_i32;
//     size_t vdims = (uint32_t)vec_dim_i32;
//     //dim ---- is actually the k to return
//     diskann::cout << "Metadata:  query #pts = " << npts << ", #dims = " << dim << ", #vec_dims = " << vdims << "... " << std::endl;

//     int truthset_type = -1; // 1 means truthset has values and distances, 2 means only ids, -1 is error

//     size_t expected_file_size_with_dists = npts * dim * vdims * sizeof(T) + npts * dim * sizeof(float) + 3 * sizeof(int); //vecs + dis + metadata

//     if (actual_file_size == expected_file_size_with_dists)
//         truthset_type = 1;

//     size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

//     if (actual_file_size == expected_file_size_just_ids)
//         truthset_type = 2;

//     if (truthset_type == -1)
//     {
//         std::stringstream stream;
//         stream << "Error. File size mismatch. File should have bin format, with "
//                   "npts followed by ngt followed by npts*ngt ids and optionally "
//                   "followed by npts*ngt distance values; actual size: "
//                << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
//                << expected_file_size_just_ids;
//         diskann::cout << stream.str();
//         throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
//     }

//     //ids = new uint32_t[npts * dim];
//     //reader.read((char *)ids, npts * dim * sizeof(uint32_t));
//     T *gt_values_flatten = new T[npts * dim * vdims];
//     reader.read((char *)gt_values_flatten, npts * dim * vdims * sizeof(T));

//     //covert the flatten to 3D vector -- std::vector<std::vector<std::vector<T>>> gt_values(query_num);
//     for (size_t i = 0; i < npts; i++) {
//         gt_values[i].resize(dim); //  resizes the i-th vector to have dim elements, each of which is an empty vector of float.
//         for (size_t j = 0; j < dim; j++) {
//             gt_values[i][j].resize(vdims); // resizes the j-th vector within the i-th vector to have vdims elements, each initialized to 0.
//             for (size_t d = 0; d < vdims; d++) {
//                 gt_values[i][j][d] = gt_values_flatten[(i * dim * vdims) + (j * vdims) + d];//replace 0 with real value
//             }
//         }
//     }

//     if (truthset_type == 1)
//     {
//         dists = new float[npts * dim];
//         reader.read((char *)dists, npts * dim * sizeof(float));
//     }

//     delete[] gt_values_flatten;
// }

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, base_file, query_file, gt_file, tags_file;
    uint64_t K;

    try
    {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");

        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("base_file", po::value<std::string>(&base_file)->required(),
                           "File containing the base vectors in binary format");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "File containing the query vectors in binary format");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                           "File name for the writing ground truth in binary "
                           "format, please don' append .bin at end if "
                           "no filter_label or filter_label_file is provided it "
                           "will save the file with '.bin' at end."
                           "else it will save the file as filename_label.bin");
        desc.add_options()("K", po::value<uint64_t>(&K)->required(),
                           "Number of ground truth nearest neighbors to compute");
        desc.add_options()("tags_file", po::value<std::string>(&tags_file)->default_value(std::string()),
                           "File containing the tags in binary format");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    if (data_type != std::string("float") && data_type != std::string("int8") && data_type != std::string("uint8"))
    {
        std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("float"))
            aux_main<float>(base_file, query_file, gt_file, K, metric, tags_file);
        if (data_type == std::string("int8"))
            aux_main<int8_t>(base_file, query_file, gt_file, K, metric, tags_file);
        if (data_type == std::string("uint8"))
            aux_main<uint8_t>(base_file, query_file, gt_file, K, metric, tags_file);
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Compute GT failed." << std::endl;
        return -1;
    }
}
