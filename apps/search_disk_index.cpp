// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix, const std::string &pq_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_pages_to_cache, const uint32_t search_io_limit, const uint32_t rerank_num,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false, const bool generate_top_nodes_list = false, const bool use_lsh = false, const bool use_subset_lsh = false, const uint32_t radius = 0)
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

//We dont have warmup query yet
    std::string warmup_query_file = "";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    // if (!query_filters.empty())
    // {
    //     filtered_search = true;
    //     if (query_filters.size() != 1 && query_filters.size() != query_num)
    //     {
    //         std::cout << "Error. Mismatch in number of queries and size of query "
    //                      "filters file"
    //                   << std::endl;
    //         return -1; // To return -1 or some other error handling?
    //     }
    // }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        ///we need consider textId only if we want to save the results into file at the end
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> dataReader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    dataReader.reset(new WindowsAlignedFileReader());
#else
    dataReader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    dataReader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(dataReader, metric));
       
//checked -- reader is also open with the finalPageGraphIndex file here
    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix, use_lsh, use_subset_lsh, radius);
    //std::cout << "LSH search radius: " << radius << std::endl;

    if (res != 0)
    {
        return res;
    }

    std::vector<uint32_t> page_list;
    diskann::cout << "Caching " << num_pages_to_cache << " most frequent visited pages based on sample data." << std::endl;
//checked: it just assign values to page_list, not generating/assigning all values to fields of this class instance
    // _pFlashIndex->cache_bfs_levels(num_pages_to_cache, page_list);
///MARK:we need this not just for mark those node for caching, but also load all PQ values of all nodes into RAM
    //_pFlashIndex->cache_bfs_levels(0, page_list);

///MARK: if there is a sample query file, we will count the frequency of visited nodes and cache only the top n most frequently visited nodes
    //std::string pageANN_warmup_query_file = "query.public.10K.u8bin";
    std::string pageANN_warmup_query_file = index_path_prefix + "_sample_data.bin";
    if (num_pages_to_cache > 0)
        //15 is searching list size; 6 is beamwidth
        _pFlashIndex->generate_cache_list_from_sample_queries(pageANN_warmup_query_file, Lvec[0], 8, num_pages_to_cache, num_threads, page_list, use_lsh);

    //checked: read the node in the list from disk and store them in _nhood_cache_buf and _coord_cache_buf
///MARK:Recover this if need caching    
    _pFlashIndex->load_cache_list(page_list);

    page_list.clear();
    page_list.shrink_to_fit(); //When you call shrink_to_fit, the vector attempts to reduce its capacity to match its size. This means it will free up any unused memory that was allocated for future growth.

    omp_set_max_active_levels(2);  // Allow 2 levels of nested parallelism
    omp_set_num_threads(num_threads);
 
    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;
//default of warmup is false
    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            //updated the parameter here as well
            _pFlashIndex->page_search(warmup + (i * warmup_aligned_dim), 1, warmup_L, 0, 
                                             warmup_result_ids.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }//end of warming up

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << "L " << std::setw(16) << "Beamwidth" << std::setw(10) << "QPS" << std::setw(10)
                  << "Mean Latency" << std::setw(10) << "Mean IOs" << std::setw(10)
                  << "Mean IO (us)" << std::setw(10) << "Hops" ;
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(10) << recall_string << std::setw(10) << "Recall@10" << std::setw(10) << "Recall@5" << std::setw(10) << "Recall@2" << std::setw(10) << "Recall@1" << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "==============================================================="
                     "======================================================="
                  << std::endl;
//buffers used to store the results
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (L < rerank_num)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than rerank_num:" << rerank_num << std::endl;
            continue;
        }

        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;
        
        query_result_ids[test_id].resize(recall_at * query_num);
        //std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        auto stats = new diskann::QueryStats[query_num];
        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();
        
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        //for (int64_t i = 0; i < (int64_t)1; i++)
        {
            if (!filtered_search)
            {
//MARK: here is where search is done -- the results are stored in query_result_ids and query_result_dists
                //std::cout << "This is query "<< i << std::endl;
#ifdef _WINDOWS                
                _pFlashIndex->page_search(query + (i * query_aligned_dim), recall_at, L, rerank_num, 
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i, use_lsh);
#else 
                //std::cout << "Use pipeline "<< i << std::endl;
                _pFlashIndex->pipelined_page_search(query + (i * query_aligned_dim), recall_at, L, rerank_num,
                                          query_result_ids_64.data() + (i * recall_at),
                                          query_result_dists[test_id].data() + (i * recall_at),
                                          optimized_beamwidth, use_reorder_data, stats + i, use_lsh);
#endif
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(), query_num, recall_at);

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_hops = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_hops; });

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num, [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto io_999 = diskann::get_percentile_stats<float>(
                    stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_cache_hits = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_cache_hits; });

        auto mean_lsh_entry_points = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_lsh_entry_points; });

        auto mean_nnbr_explored = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.nnbr_explored; });
        double recall = 0;
        double recall10 = 0;
        double recall5 = 0;
        double recall2 = 0;
        double recall1 = 0;
        if (calc_recall_flag)
        {   
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, 100);
            recall10 = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, 10);     
            recall5 = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, 5); 
            recall2 = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, 2);  
            recall1 = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, 1);                 
            best_recall = std::max(recall, best_recall);
        }
        
        diskann::cout << mean_cache_hits << std::endl;

        diskann::cout << mean_lsh_entry_points << std::endl;

        diskann::cout << mean_nnbr_explored << std::endl;
        
        diskann::cout << L << std::setw(12) << optimized_beamwidth << std::setw(10) << qps
                      << std::setw(10) << mean_latency << std::setw(10) << mean_ios << std::setw(10) << mean_io_us
                      << std::setw(10) << mean_hops;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(10) << recall << std::setw(10) << recall10 << std::setw(10) << recall5 << std::setw(10) << recall2 << std::setw(10) << recall1 << std::endl;
        }
        else
            diskann::cout << std::endl;
        delete[] stats;

        // // Map to store the count of each page
        // std::unordered_map<uint32_t, uint32_t> pageCount;

        // // Iterate through each vector of every query in intersectedVisitedPages
        // for (const auto& pages : intersectedVisitedPages) {
        //     for (const auto& page : pages) {
        //         pageCount[page]++;
        //     }
        // }

        // std::ofstream outputFile("pageCounts.txt");

        // // Write the counts to the file
        // for (const auto& [page, count] : pageCount) {
        //     outputFile << "Page " << page << ": " << count << " times" << std::endl;
        // }

        // // Close the file
        // outputFile.close();

    }

    diskann::cout << "Done searching. Not save results " << std::endl;
    // uint64_t test_id = 0;
    // for (auto L : Lvec)
    // {
    //     if (L < recall_at)
    //         continue;

    //     std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    //     diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

    //     cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    //     diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    // }

    diskann::aligned_free(query);
    if (gt_ids != nullptr) 
        delete[] gt_ids;
    if (gt_dists != nullptr) 
        delete[] gt_dists;
    if (warmup != nullptr)
        diskann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, pq_path_prefix, filter_label,
        label_type, query_filters_file, use_lsh_str, use_subset_lsh_str, generate_top_nodes_list_str;
    uint32_t num_threads, K, W, radius, num_pages_to_cache, search_io_limit, rerank_num;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;
    bool use_lsh = false;
    bool use_subset_lsh = false;
    bool generate_top_nodes_list = false;

    po::options_description desc{
        program_options_utils::make_program_description("search_disk_index", "Searches on-disk DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters 
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("pq_path_prefix", po::value<std::string>(&pq_path_prefix)->default_value(std::string("")),
                                       "Path for PQ data");
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_pages_to_cache", po::value<uint32_t>(&num_pages_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()("search_io_limit", po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
                                        "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.  Default value: false");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("use_lsh", po::value<std::string>(&use_lsh_str)->default_value(std::string("false")), "Enable LSH in searching.");
        optional_configs.add_options()("use_subset_lsh", po::value<std::string>(&use_subset_lsh_str)->default_value(std::string("false")), "Use subset of base LSH in searching.");
        optional_configs.add_options()("radius", po::value<uint32_t>(&radius)->default_value(0), "Radius of LSH searching.");
        optional_configs.add_options()("generate_top_nodes_list", po::value<std::string>(&generate_top_nodes_list_str)->default_value(std::string("false")), "If generate the list of most frequently visited nodes");
        optional_configs.add_options()("rerank_num", po::value<uint32_t>(&rerank_num)->default_value(0), "Number to rereank.");
        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;

        //use_lsh = vm["use_lsh"].as<bool>();
        if (use_lsh_str == "true" || use_lsh_str == "True")
            use_lsh = true;

        if (use_subset_lsh_str == "true" || use_subset_lsh_str == "True")
            use_subset_lsh = true;
        
        if (generate_top_nodes_list_str == "true" || generate_top_nodes_list_str == "True")
            generate_top_nodes_list = true;
        // Print to verify correctness
        std::cout << "Use LSH: " << std::boolalpha << use_lsh << std::endl;
        std::cout << "Use a subset of base LSH: " << std::boolalpha << use_subset_lsh << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("float"))
                return search_disk_index<float, uint16_t>(
                    metric, index_path_prefix, pq_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, rerank_num, Lvec, fail_if_recall_below, query_filters, use_reorder_data, generate_top_nodes_list, use_lsh, use_subset_lsh, radius);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, pq_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, rerank_num, Lvec, fail_if_recall_below, query_filters, use_reorder_data, generate_top_nodes_list, use_lsh, use_subset_lsh, radius);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, pq_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_pages_to_cache, search_io_limit, rerank_num, Lvec, fail_if_recall_below, query_filters, use_reorder_data, generate_top_nodes_list, use_lsh, use_subset_lsh, radius);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
                return search_disk_index<float>(metric, index_path_prefix, pq_path_prefix, result_path_prefix, query_file, gt_file,
                                                num_threads, K, W, num_pages_to_cache, search_io_limit, rerank_num, Lvec,
                                                fail_if_recall_below, query_filters, use_reorder_data, generate_top_nodes_list, use_lsh, use_subset_lsh, radius);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, pq_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_pages_to_cache, search_io_limit, rerank_num, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data, generate_top_nodes_list, use_lsh, use_subset_lsh, radius);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, pq_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_pages_to_cache, search_io_limit, rerank_num, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data, generate_top_nodes_list, use_lsh, use_subset_lsh, radius);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}