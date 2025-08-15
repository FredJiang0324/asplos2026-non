// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include <string>
#include <boost/program_options.hpp>
#include "utils.h"
#include "disk_utils.h"
#include "defaults.h"
#include "program_options_utils.hpp"


namespace po = boost::program_options;

int main(int argc, char **argv)
{
    std::string index_prefix_path, data_type, dist_fn, base_data_file, use_lsh_str, full_ooc_str;
    uint32_t min_degree_per_node, num_pq_chunks_32, maxVamanaDegree;
    bool use_lsh = false;
    bool full_ooc = false;
    float memBudgetInGB = 0.0f;

    try
    {
        po::options_description desc{"Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(), "distance function");
        desc.add_options()("data_path", po::value<std::string>(&base_data_file)->required(), "File to read for base raw data");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_prefix_path)->required(), "Graph index prefix for files to read");
        desc.add_options()("min_degree_per_node, minND", po::value<uint32_t>(&min_degree_per_node)->required(), "Page graph degree.");
        desc.add_options()("num_PQ_chunks", po::value<uint32_t>(&num_pq_chunks_32)->required(), "Num PQ chunks.");
        desc.add_options()("R", po::value<uint32_t>(&maxVamanaDegree)->required(), "Max degree of input vamana index");
        desc.add_options()("use_lsh", po::value<std::string>(&use_lsh_str)->required(), "Enable LSH in searching.");
        desc.add_options()("memBudgetInGB", po::value<float>(&memBudgetInGB)->required(), "Mem Budget In GB");
        desc.add_options()("full_ooc", po::value<std::string>(&full_ooc_str)->required(), "Adopt fully out-of-core solution in searching.");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        
        if (use_lsh_str == "true" || use_lsh_str == "True")
            use_lsh = true;
        
        if (full_ooc_str == "true" || full_ooc_str == "True")
            full_ooc = true;
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
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == std::string("cosine"))
        metric = diskann::Metric::COSINE;
    else
    {
        std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("float"))
             diskann::build_page_graph<float>(index_prefix_path, base_data_file, min_degree_per_node, maxVamanaDegree, num_pq_chunks_32, metric, memBudgetInGB, use_lsh, full_ooc);
        if (data_type == std::string("int8"))
             diskann::build_page_graph<int8_t>(index_prefix_path, base_data_file, min_degree_per_node, maxVamanaDegree, num_pq_chunks_32, metric, memBudgetInGB, use_lsh, full_ooc);
        if (data_type == std::string("uint8"))
             diskann::build_page_graph<uint8_t>(index_prefix_path, base_data_file, min_degree_per_node, maxVamanaDegree, num_pq_chunks_32, metric, memBudgetInGB, use_lsh, full_ooc);
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Generate page graph failed." << std::endl;
        return -1;
    }
}
