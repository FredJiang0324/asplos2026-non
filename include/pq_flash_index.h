// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "common_includes.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "tsl/robin_map.h"
#include <tsl/sparse_map.h>
#include "tsl/robin_set.h"
#include <bitset>

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

template <typename T, typename LabelT = uint32_t> class PQFlashIndex
{
  public:
    // DISKANN_DLLEXPORT PQFlashIndex(std::shared_ptr<AlignedFileReader> &dataReader, std::shared_ptr<AlignedFileReader> &topoReader,
    //                                diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT PQFlashIndex(std::shared_ptr<AlignedFileReader> &dataReader, diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix, const std::string &pq_path_prefix, const bool use_lsh, const bool use_subset_lsh, const uint32_t radius);
#endif

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                   const char *index_filepath, const char *pivots_filepath,
                                                   const char *compressed_filepath, const bool use_lsh, const bool use_subset_lsh);
#else
    DISKANN_DLLEXPORT int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                   const char *pivots_filepath, const char *compressed_filepath, const bool use_lsh, const bool use_subset_lsh);
#endif

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &page_list);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                   uint64_t l_search, uint64_t beamwidth,
                                                                   uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                   std::vector<uint32_t> &node_list);
#else
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                    uint64_t beamwidth, uint32_t num_pages_to_cache,
                                                                    uint32_t nthreads,
                                                                    std::vector<uint32_t> &page_list, bool use_lsh);
#endif

    DISKANN_DLLEXPORT void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                            const bool shuffle = false);

    //DISKANN_DLLEXPORT void cache_all_pages_all_nodes_pqValues();

    DISKANN_DLLEXPORT void page_search(const T *query, const uint64_t k_search, const uint64_t l_search, const uint64_t rerank_num,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr, const bool use_lsh = false);

    DISKANN_DLLEXPORT void page_search(const T *query, const uint64_t k_search, const uint64_t l_search, const uint64_t rerank_num,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr, const bool use_lsh = false);

    DISKANN_DLLEXPORT void page_search(const T *query, const uint64_t k_search, const uint64_t l_search, const uint64_t rerank_num,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr, const bool use_lsh = false);

    DISKANN_DLLEXPORT void page_search(const T *query, const uint64_t k_search, const uint64_t l_search, const uint64_t rerank_num,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr, const bool use_lsh = false);
    DISKANN_DLLEXPORT void pipelined_page_search(const T *query, const uint64_t k_search, const uint64_t l_search, const uint64_t rerank_num,
                                                uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                                const bool use_reorder_data = false, QueryStats *stats = nullptr, const bool use_lsh = false);

    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &filter_label);

    DISKANN_DLLEXPORT uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                            const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                            std::vector<float> &distances, const uint64_t min_beam_width,
                                            QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT uint64_t get_data_dim();

    std::shared_ptr<AlignedFileReader> &dataReader;

    //std::shared_ptr<AlignedFileReader> &topoReader;

    DISKANN_DLLEXPORT diskann::Metric get_metric();

    //
    // node_ids: input list of node_ids to be read
    // coord_buffers: pointers to pre-allocated buffers that coords need to copied to. If null, dont copy.
    // nbr_buffers: pre-allocated buffers to copy neighbors into
    //
    // returns a vector of bool one for each node_id: true if read is success, else false
    //
    DISKANN_DLLEXPORT std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<uint32_t *> &nbr_buffers);

    DISKANN_DLLEXPORT std::vector<bool> read_page_all_nodes(const std::vector<uint32_t> &page_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<uint32_t *> &nbr_buffers);
    
    DISKANN_DLLEXPORT std::vector<bool> read_page_nbrs(const std::vector<uint32_t> &page_ids, std::vector<std::vector<uint32_t>*> &nbr_buffers);
                                                   
    DISKANN_DLLEXPORT std::vector<std::uint8_t> get_pq_vector(std::uint64_t vid);
    DISKANN_DLLEXPORT uint64_t get_num_points();

  protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);

    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

  private:
    DISKANN_DLLEXPORT inline bool point_has_label(uint32_t point_id, LabelT label_id);
    std::unordered_map<std::string, LabelT> load_label_map(std::basic_istream<char> &infile);
    DISKANN_DLLEXPORT void parse_label_file(std::basic_istream<char> &infile, size_t &num_pts_labels);
    DISKANN_DLLEXPORT void get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                   uint32_t &num_total_labels);
    DISKANN_DLLEXPORT void generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                  const uint32_t nthreads);
    void reset_stream_for_reading(std::basic_istream<char> &infile);

    // sector # on disk where node_id is present with in the graph part
    DISKANN_DLLEXPORT uint64_t get_node_sector(uint64_t node_id);

    // ptr to start of the node
    DISKANN_DLLEXPORT char *offset_to_node(char *sector_buf, uint64_t node_id);

    // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
    DISKANN_DLLEXPORT uint32_t *offset_to_node_nhood(char *node_buf);

    // returns region of `node_buf` containing [COORD(T)]
    DISKANN_DLLEXPORT T *offset_to_node_coords(char *node_buf);

    // index info for multi-node sectors
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    //
    // index info for multi-sector nodes
    // nhood of node `i` is in sector: [i * DIV_ROUND_UP(_max_node_len, SECTOR_LEN)]
    // offset in sector: [0]
    //
    // Common info
    // coords start at ofsset
    // #nbrs of node `i`: *(unsigned*) (offset + disk_bytes_per_point)
    // nbrs of node `i` : (unsigned*) (offset + disk_bytes_per_point + 1)

    uint64_t _max_node_len = 0;
    uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
    ///MARK: newly added
    //uint64_t _nnodes_per_mega_node = 0;
    // uint64_t _n_sectors_for_topo = 0;
    // uint64_t _n_sectors_for_data = 0;
    uint64_t _page_degree = 0;
    uint64_t _max_degree = 0;

    ///MARK: newly added for PQ cached data
    uint32_t _num_pq_cached_nodes = 0;
    bool _useID_pqIdx_map = false;
    bool _useID_pqIdx_array = false;
    bool _no_pq_cached = true;
    bool _all_pq_cached = false;
    uint8_t* _cached_pq_buff = nullptr;
    tsl::sparse_map<uint32_t, uint32_t> _nodeID_pqIdx_map;//cache ratio is less than 50%
    //tsl::robin_map<uint32_t, uint32_t> _nodeID_pqIdx_map;
    std::vector<uint32_t> _nodeID_pqIdx_arr; //used when more than 50% but less 100% nodes' PQ are cached

    ///MARK: DS for lsh
    tsl::sparse_map<uint32_t, std::pair<uint32_t*, uint8_t>> _buckets; //each bucket 256 max
    //tsl::robin_map<uint8_t, uint8_t> _pageOffset_pqOffset_map;
    //uint8_t* _all_IDs_PQ_LSH = nullptr; //this is where the pointers in _buckets point to
    uint32_t* _sample_nodes_IDs_in_LSH = nullptr;
    float* _projectionMatrix = nullptr;
    int _numProjections;
    //size_t _num_sampled_nodes_per_sector;
    //uint8_t *_lsh_pq_coord_cache_buf = nullptr;
    uint8_t _radius = 0;

    // Data used for searching with re-order vectors
    uint64_t _ndims_reorder_vecs = 0;
    uint64_t _reorder_data_start_sector = 0;
    uint64_t _nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float _max_base_norm = 0.0f;

    // data info
    uint64_t _num_points = 0;
    uint64_t _num_pages = 0;
    uint64_t _num_frozen_points = 0;
    uint64_t _frozen_location = 0;
    uint64_t _data_dim = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _disk_bytes_per_point = 0; // Number of bytes

    std::string _disk_index_file;
    std::vector<std::pair<uint32_t, uint32_t>> _node_visit_counter;
    std::vector<std::pair<uint32_t, uint32_t>> _page_visit_counter;
    tsl::robin_set<uint32_t> _top_hops_pages;
    //tsl::robin_map<uint32_t, std::vector<uint32_t>> _buckets;

    // PQ data
    // _n_chunks = # of chunks ndims is split into
    // data: char * _n_chunks
    // chunk_size = chunk size of each dimension chunk
    //so, in total, there are 256 (2^8) * numChuck centroids. and each centroid correspond to a float
    // pq_tables = float* [[2^8 * [chunk_size]] * _n_chunks]
    uint8_t *data = nullptr; //has manual delete
    uint64_t _n_chunks; //this is the number of PQ chunk -- size of compressed pq data
    FixedChunkPQTable _pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>> _dist_cmp;
    std::shared_ptr<Distance<float>> _dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    bool _use_disk_index_pq = false;
    uint64_t _disk_pq_n_chunks = 0;
    FixedChunkPQTable _disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *_medoids = nullptr; //has manual delete
    // defaults to 1
    size_t _num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *_centroid_data = nullptr; //has manual delete

    ///MARK: cached pages nbrs and vector values
    ///MARK: newly added:
    tsl::sparse_map<uint32_t, uint32_t> _cached_page_idx_map; //if the cache ratio is more than 1/4, we just use array instead of map
    uint32_t* _nhood_cache_buf = nullptr;//has manual delete
    T *_coord_cache_buf = nullptr; //has manual delete

    ///MARK: newly added
    //uint8_t *_pq_coord_cache_buf = nullptr; //has manual delete
    //tsl::sparse_map<uint32_t, uint8_t *> _pq_coord_cache;//no need of delete. key is the pageID while the value is pointer which points to _pq_coord_cache_buf where pq value of representative node of the page  
    // uint64_t _cached_pq_val_count = 0;
    // const uint32_t INVALID_NBR_ID = 0xFFFFFFFF;

    // thread-specific scratch
    ConcurrentQueue<SSDThreadData<T> *> _thread_data; //has manual delete
    uint64_t _max_nthreads;
    bool _load_flag = false;
    //bool _count_visited_nodes = false;
    bool _count_visited_pages = false;
    bool _getMostFrequentlyVisitedNodes = false;
    bool _reorder_data_exists = false;
    uint64_t _reoreder_data_offset = 0;

    // filter support
    uint32_t *_pts_to_label_offsets = nullptr; //has manual delete
    uint32_t *_pts_to_label_counts = nullptr; //has manual delete
    LabelT *_pts_to_labels = nullptr; //has manual delete
    std::unordered_map<LabelT, std::vector<uint32_t>> _filter_to_medoid_ids;
    bool _use_universal_label = false;
    LabelT _universal_filter_label;
    tsl::robin_set<uint32_t> _dummy_pts;
    tsl::robin_set<uint32_t> _has_dummy_pts;
    tsl::robin_map<uint32_t, uint32_t> _dummy_to_real_map;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _real_to_dummy_map;
    std::unordered_map<std::string, LabelT> _label_map;

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = defaults::SECTOR_LEN;
    char *getHeaderBytes();
#endif
};
} // namespace diskann
