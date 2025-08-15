// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ooc_in_mem_graph_store.h"
#include "utils.h"
#include "defaults.h"
#include <iomanip> 

namespace diskann
{
InMemOOCGraphStore::InMemOOCGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
    : AbstractGraphStore(total_pts, reserve_graph_degree)
{
    //_capacity = total_pts;
    _max_degree = static_cast<uint32_t>(reserve_graph_degree);
    _graph_data = new uint32_t[total_pts * reserve_graph_degree];
    _graph_nbrs_size = new uint8_t[total_pts];
}

InMemOOCGraphStore::~InMemOOCGraphStore()
{
    if (_graph_data != nullptr)
    {
        delete[] _graph_data;
    }

    if (_graph_nbrs_size != nullptr)
    {
        delete[] _graph_nbrs_size;
    }

    // if (_index_reader.is_open()) {
    //     _index_reader.close();
    // }
}

std::tuple<uint32_t, uint32_t, size_t> InMemOOCGraphStore::load(const std::string &index_path_prefix,
                                                             const size_t num_points)
{
    return load_impl(index_path_prefix, num_points);
}

int InMemOOCGraphStore::store(const std::string &index_path_prefix, const size_t num_points,
                           const size_t num_frozen_points, const uint32_t start)
{
    return 0;
}
const std::vector<location_t> &InMemOOCGraphStore::get_neighbours(const location_t lc) const
{      
    return _graph.at(lc);
}

std::vector<location_t> InMemOOCGraphStore::get_ooc_neighbours(const location_t nodeID)
{      
    // auto iter = _nodeID_idx_map.find(nodeID);
    // if (iter != _nodeID_idx_map.end()) {
        uint32_t* pos = _graph_data + nodeID * _max_degree;
        uint8_t nnbrs = *(_graph_nbrs_size + nodeID);
        return std::vector<location_t>(pos, pos + nnbrs);
    //}   
    
    // // Locate the data in base data file
    // size_t pageID = nodeID / _nnodes_per_sector;
    // size_t offset_within_page = nodeID % _nnodes_per_sector;
    // size_t nbrs_offset = (pageID + 1) * defaults::SECTOR_LEN + offset_within_page * _max_node_len + _node_len;
    // this->_index_reader.seekg(nbrs_offset);
    // if (!this->_index_reader) {
    //     std::cerr << "Failed to seek to file when getting the nbrs.\n";
    //     return {};
    // }

    // // Read node data from file into memory (at slot _idx_slot_to_discard)
    // size_t size_to_read = sizeof(uint32_t) + sizeof(uint32_t) * _max_degree;
    // std::unique_ptr<char[]> nbrs_buf = std::make_unique<char[]>(size_to_read);
    // this->_index_reader.read(reinterpret_cast<char*>(nbrs_buf.get()), size_to_read);
    // if (!this->_index_reader) {
    //     std::cerr << "Failed to read data for nbrs of node " << nodeID << ".\n";
    //     return {};
    // }

    // //update the data in the correspondign position in nbr and nnbr buffer
    // uint32_t raw_nnbrs = *reinterpret_cast<uint32_t*>(nbrs_buf.get());
    // ///MARK: only work for nnbr not exceed 255
    // uint8_t nnbrs = static_cast<uint8_t>(raw_nnbrs);
    // if (nnbrs > _max_degree) {
    //     std::cerr << "Warning: nnbrs (" << static_cast<int>(nnbrs) << ") exceeds _max_degree (" << _max_degree << ").\n";
    //     return {};
    // }

    // uint32_t pos_to_add = static_cast<uint32_t>(_cached_nodeIDs.size());
    // //if there is still space, not evit
    // if (pos_to_add < this->capacity()){
    //     _nodeID_idx_map[nodeID] =  pos_to_add;
    //     _cached_nodeIDs.push_back(nodeID);
    // }
    // //evict
    // else{
    //     pos_to_add = _idx_slot_to_discard;
    //     // Update map with new node and evict old one
    //     uint32_t node_to_discard = _cached_nodeIDs[_idx_slot_to_discard];
    //     _nodeID_idx_map.erase(node_to_discard); // Remove old cached node from map
    //     _nodeID_idx_map[nodeID] = pos_to_add; // Insert current node
    //     _cached_nodeIDs[pos_to_add] = nodeID;

    //     // Move to next slot for future eviction
    //     _idx_slot_to_discard = (_idx_slot_to_discard + 1) % this->capacity();
    //     _shrink_counter++;
    //     _cache_miss++;

    //     if (_shrink_counter >= 100000) {
    //         tsl::robin_map<uint32_t, uint32_t>().swap(_nodeID_idx_map);
    //         _shrink_counter = 0;
    //     }
    // }
    // //save data in the right pos
    // *(_graph_nbrs_size + pos_to_add) = nnbrs;
    // uint32_t* pos = _graph_data + pos_to_add * _max_degree;
    // memcpy(pos, nbrs_buf.get() + sizeof(uint32_t), sizeof(uint32_t) * nnbrs);
    // //return the copy of data
    // return std::vector<location_t>(pos, pos + nnbrs);
}

uint32_t InMemOOCGraphStore::get_start(){
    return _start;
}

void InMemOOCGraphStore::increase_capacity(size_t new_capacity){
    // _idx_slot_to_discard = static_cast<uint32_t>(this->capacity());
    // this->set_capacity(new_capacity);
}

void InMemOOCGraphStore::add_neighbour(const location_t i, location_t neighbour_id)
{
}

void InMemOOCGraphStore::clear_neighbours(const location_t i)
{
};
void InMemOOCGraphStore::swap_neighbours(const location_t a, location_t b)
{
};

void InMemOOCGraphStore::set_neighbours(const location_t i, std::vector<location_t> &neighbours)
{
}

size_t InMemOOCGraphStore::resize_graph(const size_t new_size)
{
    // _graph.resize(new_size);
    // set_total_points(new_size);
    //return _graph.size();
    return 0;
}

void InMemOOCGraphStore::clear_graph()
{
}

#ifdef EXEC_ENV_OLS
std::tuple<uint32_t, uint32_t, size_t> InMemOOCGraphStore::load_impl(AlignedFileReader &reader, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;

    auto max_points = get_max_points();
    int header_size = 2 * sizeof(size_t) + 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
    read_array(reader, header.get(), header_size);

    expected_file_size = *((size_t *)header.get());
    _max_observed_degree = *((uint32_t *)(header.get() + sizeof(size_t)));
    start = *((uint32_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t)));
    file_frozen_pts = *((size_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t)));

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    diskann::cout << "Loading vamana graph from reader..." << std::flush;

    // If user provides more points than max_points
    // resize the _graph to the larger size.
    if (get_total_points() < expected_num_points)
    {
        diskann::cout << "resizing graph to " << expected_num_points << std::endl;
        this->resize_graph(expected_num_points);
    }

    uint32_t nodes_read = 0;
    size_t cc = 0;
    size_t graph_offset = header_size;
    while (nodes_read < expected_num_points)
    {
        uint32_t k;
        read_value(reader, k, graph_offset);
        graph_offset += sizeof(uint32_t);
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        read_array(reader, tmp.data(), k, graph_offset);
        graph_offset += k * sizeof(uint32_t);
        cc += k;
        _graph[nodes_read].swap(tmp);
        nodes_read++;
        if (nodes_read % 1000000 == 0)
        {
            diskann::cout << "." << std::flush;
        }
        if (k > _max_range_of_graph)
        {
            _max_range_of_graph = k;
        }
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to " << start
                  << std::endl;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}
#endif

std::tuple<uint32_t, uint32_t, size_t> InMemOOCGraphStore::load_impl(const std::string &filename,
                                                                  size_t target_num_points)
{
    size_t file_offset = 0; // will need this for single file format support
    std::ifstream index_reader(filename, std::ios::binary);
    if (!index_reader.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    index_reader.seekg(file_offset);

    diskann::cout << "Loading vamana index graph's metadata from " << filename << "..." << std::endl;
    uint32_t nr, nc;
    uint64_t npts_64, ndims_64, medoid, max_node_len, nnodes_per_sector;
    index_reader.read((char *)&nr, sizeof(uint32_t));
    index_reader.read((char *)&nc, sizeof(uint32_t));
    index_reader.read((char *)&npts_64, sizeof(uint64_t));
    index_reader.read((char *)&ndims_64, sizeof(uint64_t));
    index_reader.read((char *)&medoid, sizeof(uint64_t));
    index_reader.read((char *)&max_node_len, sizeof(uint64_t));
    index_reader.read((char *)&nnodes_per_sector, sizeof(uint64_t));

    _start = static_cast<uint32_t>(medoid);
    //_max_node_len = static_cast<uint32_t>(max_node_len);
    _nnodes_per_sector = static_cast<uint32_t>(nnodes_per_sector);
    _node_len = ndims_64 * _type_size;
    //_max_degree = _max_observed_degree;

    diskann::cout << "Loading vamana index graph from " << filename << "..." << std::endl;
    diskann::cout << "nr: " << nr << std::endl;
    diskann::cout << "nc: " << nc << std::endl;
    diskann::cout << "npts_64: " << npts_64 << std::endl;
    diskann::cout << "ndims_64: " << ndims_64 << std::endl;
    diskann::cout << "medoid: " << medoid << std::endl;
    //diskann::cout << "max node len: " << max_node_len << std::endl;
    //_nodeID_idx_map.reserve(this->capacity());
    //_cached_nodeIDs.reserve(this->capacity());

    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(defaults::SECTOR_LEN);
    uint32_t nnbrs = 0;
    uint32_t num_page_to_read = (static_cast<uint32_t>(target_num_points) + _nnodes_per_sector - 1) / _nnodes_per_sector;
    diskann::cout << "Number of sectors (pages) to read: " << num_page_to_read << std::endl;
    uint32_t nnodes_cached = 0;
    uint32_t max_degree = 0;
    uint32_t min_degree = std::numeric_limits<uint32_t>::max();
    uint64_t total_degree = 0;
    for (size_t i = 0; i < num_page_to_read; i++){
        if (i % 100000 == 0 || i == num_page_to_read - 1) {
            float percent = 100.0f * (i + 1) / num_page_to_read;
            diskann::cout << "\rProgress: " << std::setw(6) << std::fixed << std::setprecision(2)
                        << percent << "% (" << (i + 1) << "/" << num_page_to_read << ")"
                        << std::flush;
        }
        //read a whole page at a time // Skip metadata sector at offset 0
        file_offset = (i + 1) * defaults::SECTOR_LEN;
        index_reader.seekg(file_offset);
        //memset(sector_buf.get(), 0, defaults::SECTOR_LEN);
        index_reader.read(reinterpret_cast<char*>(sector_buf.get()), defaults::SECTOR_LEN);
        
        //decode the data of this page
        for (size_t k = 0; k < _nnodes_per_sector; k++){
            auto node_nnbr_ptr = reinterpret_cast<uint32_t*>(sector_buf.get() + k * max_node_len + _node_len);
            nnbrs = *node_nnbr_ptr;
            total_degree += (uint64_t)nnbrs;
            max_degree = std::max(max_degree, nnbrs);
            min_degree = std::min(min_degree, nnbrs);
           
            *(_graph_nbrs_size + nnodes_cached) = static_cast<uint8_t>(nnbrs);
            memcpy(reinterpret_cast<char*>(_graph_data + nnodes_cached * _max_degree), node_nnbr_ptr + 1, nnbrs * sizeof(uint32_t));
            // _nodeID_idx_map.emplace(nnodes_cached, nnodes_cached);
            // _cached_nodeIDs.push_back(nnodes_cached);

            nnodes_cached++;
            if (nnodes_cached >= target_num_points){
                break;
            }
        }
    }
    index_reader.close();
    //diskann::cout << std::endl;
    diskann::cout << "\nMax degree: " << max_degree << std::endl;
    diskann::cout << "Min degree: " << min_degree << std::endl;
    diskann::cout << "Mean degree: " << static_cast<double>(total_degree) / nnodes_cached << std::endl;
    diskann::cout << "Done. Index has cached " << nnodes_cached << " nodes" << std::endl;
    diskann::cout << std::endl;
    return std::make_tuple(nnodes_cached, _start, 0);
}

void InMemOOCGraphStore::set_type_size(size_t size){
    _type_size = size;
}

int InMemOOCGraphStore::save_graph(const std::string &index_path_prefix, const size_t num_points,
                                const size_t num_frozen_points, const uint32_t start)
{
    return 0;
}

size_t InMemOOCGraphStore::getGraphNodesNum() const {
    return 0;
}

size_t InMemOOCGraphStore::get_max_range_of_graph()
{
    return _max_range_of_graph;
}

uint32_t InMemOOCGraphStore::get_max_observed_degree()
{
    return _max_observed_degree;
}

} // namespace diskann
