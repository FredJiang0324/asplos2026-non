// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <memory>
#include "abstract_scratch.h"
#include "ooc_in_mem_data_store.h"
#include <iomanip> 
#include "utils.h"

namespace diskann
{

template <typename data_t>
InMemOOCDataStore<data_t>::InMemOOCDataStore(const location_t num_points, const size_t dim,
                                       std::unique_ptr<Distance<data_t>> distance_fn)
    : AbstractDataStore<data_t>(num_points, dim), _distance_fn(std::move(distance_fn))
{
    //_distance_fn->get_required_alignment() will return 8 //for instance, for dim of 123, its -aligned_dim will be 128
    _aligned_dim = ROUND_UP(dim, _distance_fn->get_required_alignment());
    alloc_aligned(((void **)&_data), this->_capacity * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    std::memset(_data, 0, this->_capacity * _aligned_dim * sizeof(data_t));
}

template <typename data_t> InMemOOCDataStore<data_t>::~InMemOOCDataStore()
{
    if (_data != nullptr)
    {
        aligned_free(this->_data);
    }

    // if (_base_reader.is_open()) {
    //     _base_reader.close();
    // }
}

template <typename data_t> size_t InMemOOCDataStore<data_t>::get_aligned_dim() const
{
    return _aligned_dim;
}

template <typename data_t> size_t InMemOOCDataStore<data_t>::get_alignment_factor() const
{
    return _distance_fn->get_required_alignment();
}

template <typename data_t> location_t InMemOOCDataStore<data_t>::load(const std::string &filename)
{
    return load_impl(filename);
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t InMemOOCDataStore<data_t>::load_impl(AlignedFileReader &reader)
{
    size_t file_dim, file_num_points;

    diskann::get_bin_metadata(reader, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->capacity())
    {
        this->resize((location_t)file_num_points);
    }
    copy_aligned_data_from_file<data_t>(reader, _data, file_num_points, file_dim, _aligned_dim);

    return (location_t)file_num_points;
}
#endif

template <typename data_t> location_t InMemOOCDataStore<data_t>::load_impl(const std::string &filename)
{
    size_t file_dim, file_num_points;
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "data_file_num_points: " << file_num_points << std::endl;
    diskann::cout << "data_file_dim: " << file_dim << std::endl;

    // if (file_num_points > this->capacity())
    // {
    //     this->resize((location_t)file_num_points);
    // }
    //size_t npts_to_cache = static_cast<size_t>(this->capacity());
    diskann::cout << "npts_to_cache: " << this->capacity() << std::endl << std::endl;
    copy_aligned_data_from_file<data_t>(filename.c_str(), _data, file_num_points, file_dim, _aligned_dim);
    diskann::cout << "Finish loading data points to _data " << std::endl;

    // this->_base_reader.open(filename, std::ios::binary);
    // //diskann::cout << "Open _base_reader file " << std::endl;
    // if (!this->_base_reader) {
    //     throw std::runtime_error("Failed to open file: " + filename);
    // }

    // this->_node_len = static_cast<uint32_t>(this->_dim * sizeof(data_t));

    // this->_metadata_size = 2 * sizeof(int);

    // _nodeID_idx_map.reserve(this->capacity());
    // _cached_nodeIDs.reserve(this->capacity());

    // for(uint32_t i = 0; i < this->capacity(); i++){
    //     //_nodeID_idx_map[i] = i;
    //     //_cached_nodeIDs.push_back(-1);
    // }

    return (location_t)this->capacity();
}

template <typename data_t> size_t InMemOOCDataStore<data_t>::save(const std::string &filename, const location_t num_points)
{
    return save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

template <typename data_t> void InMemOOCDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    memset(_data, 0, _aligned_dim * sizeof(data_t) * num_pts);
    for (location_t i = 0; i < num_pts; i++)
    {
        std::memmove(_data + i * _aligned_dim, vectors + i * this->_dim, this->_dim * sizeof(data_t));
    }

    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data, this->_aligned_dim, num_pts);
    }
}

template <typename data_t> void InMemOOCDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    size_t npts, ndim;
    copy_aligned_data_from_file(filename.c_str(), _data, npts, ndim, _aligned_dim, offset);

    if ((location_t)npts > this->capacity())
    {
        std::stringstream ss;
        ss << "Number of points in the file: " << filename
           << " is greater than the capacity of data store: " << this->capacity()
           << ". Must invoke resize before calling populate_data()" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    if ((location_t)ndim != this->get_dims())
    {
        std::stringstream ss;
        ss << "Number of dimensions of a point in the file: " << filename
           << " is not equal to dimensions of data store: " << this->capacity() << "." << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    if (_distance_fn->preprocessing_required())
    {
         diskann::cout << "Preprocess (normalize) data when populating data from file" << std::endl;
        _distance_fn->preprocess_base_points(_data, this->_aligned_dim, this->capacity());
    }
}

template <typename data_t>
void InMemOOCDataStore<data_t>::extract_data_to_bin(const std::string &filename, const location_t num_points)
{
    save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

//this is only used in create_disk_layout to insert vector into the page --- hence, no need to cache it anymore. instead, if find it, we should clear its memory to save space
///MARK: it more worthy to use space for storing graph info -- we dont need _data buffer anymore
template <typename data_t> void InMemOOCDataStore<data_t>::get_vector(const location_t i, data_t *dest) const
{
    // REFACTOR TODO: Should we denormalize and return values?
    ///MARK: reading the node from the diskANN_index and not saving
    // auto iter = _nodeID_idx_map.find(i);
    // if (iter != _nodeID_idx_map.end()) {
        memcpy(dest, _data + i * _aligned_dim, this->_dim * sizeof(data_t));
        //return;
    //}   

    // Compute offset
    // size_t offset = this->_metadata_size + i * this->_node_len;
    // this->_base_reader.seekg(offset);
    // if (!this->_base_reader) {
    //     std::cerr << "Failed to seek to page when getting the vector.\n";
    //     return;
    // }
    // this->_base_reader.read(reinterpret_cast<char*>(dest), this->_node_len);
    //memcpy(dest, _data + lc * _aligned_dim, this->_dim * sizeof(data_t));
}

template <typename data_t> std::vector<float> InMemOOCDataStore<data_t>::computeDist(uint32_t node, const std::vector<location_t>& neighbors) {
    ///MARK: check and make sure node and all neighbors are avalible
    //look_cache_node(node);
    std::vector<float> distances;
    for (const auto& neighbor : neighbors) {
        //look_cache_node(static_cast<uint32_t>(neighbor));
        distances.push_back(get_distance(node, neighbor));
    }

    // if (_shrink_counter >= 100000) {
    //     tsl::robin_map<uint32_t, uint32_t>().swap(_nodeID_idx_map);
    //     _shrink_counter = 0;
    // }
    return distances;
}

template <typename data_t> void InMemOOCDataStore<data_t>::look_cache_node(uint32_t i){
    // if (_nodeID_idx_map.find(i) != _nodeID_idx_map.end()) return;
    
    // // Locate the data in base data file
    // size_t offset = this->_metadata_size + i * this->_node_len;
    // this->_base_reader.seekg(offset);
    // if (!this->_base_reader) {
    //     std::cerr << "Failed to seek to page when getting the vector.\n";
    //     return;
    // }

    // //if there is still space, not evit
    // uint32_t pos_to_add = static_cast<uint32_t>(_cached_nodeIDs.size());
    // if (pos_to_add < this->capacity()){
    //     _nodeID_idx_map[i] =  pos_to_add;
    //     _cached_nodeIDs.push_back(i);
    // }
    // //evict
    // else{
    //     pos_to_add = _idx_slot_to_discard;
    //     uint32_t node_to_discard = _cached_nodeIDs[_idx_slot_to_discard];
    //     _nodeID_idx_map.erase(node_to_discard);

    //     _nodeID_idx_map[i] = _idx_slot_to_discard;
    //     _cached_nodeIDs[_idx_slot_to_discard] = i;

    //     // Move to next slot for future eviction
    //     _idx_slot_to_discard = (_idx_slot_to_discard + 1) % this->capacity();
    //     _shrink_counter++;
    //     _cache_miss++;
    // }

    // // Read node data from file into memory (at slot _idx_slot_to_discard)
    // this->_base_reader.read(reinterpret_cast<char*>(_data + pos_to_add * _aligned_dim), this->_node_len);
    // if (!this->_base_reader) {
    //     std::cerr << "Failed to read data for node " << i << ".\n";
    //     return;
    // }
    //diskann::cout << "finish check/cache the node in memory" << std::endl;

}

template <typename data_t> void InMemOOCDataStore<data_t>::cleanup_cache(){
    // if (_data != nullptr)
    // {
    //     aligned_free(this->_data);
    // }
    // _nodeID_idx_map.clear();
    // tsl::robin_map<uint32_t, uint32_t>().swap(_nodeID_idx_map); // releases memory back to the system
    // _cached_nodeIDs.clear();
    // _cached_nodeIDs.shrink_to_fit(); // standard trick to reduce vector's capacity
}

template <typename data_t> void InMemOOCDataStore<data_t>::set_nnodes_to_load(size_t n){
    //_nnodes_to_load = n;
}

template <typename data_t> void InMemOOCDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    size_t offset_in_data = loc * _aligned_dim;
    memset(_data + offset_in_data, 0, _aligned_dim * sizeof(data_t));
    memcpy(_data + offset_in_data, vector, this->_dim * sizeof(data_t));
    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data + offset_in_data, _aligned_dim, 1);
    }
}

template <typename data_t> void InMemOOCDataStore<data_t>::prefetch_vector(const location_t loc)
{
    diskann::prefetch_vector((const char *)_data + _aligned_dim * (size_t)loc * sizeof(data_t),
                             sizeof(data_t) * _aligned_dim);
}

template <typename data_t>
void InMemOOCDataStore<data_t>::preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const
{
    if (query_scratch != nullptr)
    {
        memcpy(query_scratch->aligned_query_T(), query, sizeof(data_t) * this->get_dims());
    }
    else
    {
        std::stringstream ss;
        ss << "In InMemOOCDataStore::preprocess_query: Query scratch is null";
        diskann::cerr << ss.str() << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
}

template <typename data_t> float InMemOOCDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    return _distance_fn->compare(query, _data + _aligned_dim * loc, (uint32_t)_aligned_dim);
}

template <typename data_t>
void InMemOOCDataStore<data_t>::get_distance(const data_t *query, const location_t *locations,
                                          const uint32_t location_count, float *distances,
                                          AbstractScratch<data_t> *scratch_space) const
{
    for (location_t i = 0; i < location_count; i++)
    {
        distances[i] = _distance_fn->compare(query, _data + locations[i] * _aligned_dim, (uint32_t)this->_aligned_dim);
    }
}

//this is the one we use
template <typename data_t>
float InMemOOCDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    //diskann::cout << "begin to get distance bwt " << loc1 << " and " << loc2 << std::endl;
    // auto iter1 = _nodeID_idx_map.find(loc1);
    // auto iter2 = _nodeID_idx_map.find(loc2);
    // uint8_t * pos1 = reinterpret_cast<uint8_t*>(_data + iter1->second * _aligned_dim);
    // uint8_t * pos2 = reinterpret_cast<uint8_t*>(_data + iter2->second * _aligned_dim);
    // for (int32_t i = 0; i < (int32_t)this->_aligned_dim; i++)
    // {
    //     std::cout << static_cast<int>(pos1[i]) << " " << static_cast<int>(pos2[i]) << std::endl;
    // }

    return _distance_fn->compare(_data + loc1 * _aligned_dim, _data + loc2 * _aligned_dim,
                                (uint32_t)this->_aligned_dim);
}

template <typename data_t>
void InMemOOCDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                          std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const
{
    for (int i = 0; i < ids.size(); i++)
    {
        distances[i] =
            _distance_fn->compare(preprocessed_query, _data + ids[i] * _aligned_dim, (uint32_t)this->_aligned_dim);
    }
}

template <typename data_t> location_t InMemOOCDataStore<data_t>::expand(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size < this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'expand' datastore when new capacity (" << new_size << ") < existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _data, this->capacity() * _aligned_dim * sizeof(data_t));
    aligned_free(_data);
    _data = new_data;
#else
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

template <typename data_t> location_t InMemOOCDataStore<data_t>::shrink(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size > this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'shrink' datastore when new capacity (" << new_size << ") > existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _data, new_size * _aligned_dim * sizeof(data_t));
    aligned_free(_data);
    _data = new_data;
#else
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

template <typename data_t>
void InMemOOCDataStore<data_t>::move_vectors(const location_t old_location_start, const location_t new_location_start,
                                          const location_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    /*    // Update pointers to the moved nodes. Note: the computation is correct
       even
        // when new_location_start < old_location_start given the C++ uint32_t
        // integer arithmetic rules.
        const uint32_t location_delta = new_location_start - old_location_start;
    */
    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    if (new_location_start < old_location_start)
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }

    // Use memmove to handle overlapping ranges.
    copy_vectors(old_location_start, new_location_start, num_locations);
    memset(_data + _aligned_dim * mem_clear_loc_start, 0,
           sizeof(data_t) * _aligned_dim * (mem_clear_loc_end_limit - mem_clear_loc_start));
}

template <typename data_t>
void InMemOOCDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc,
                                          const location_t num_points)
{
    assert(from_loc < this->_capacity);
    assert(to_loc < this->_capacity);
    assert(num_points < this->_capacity);
    memmove(_data + _aligned_dim * to_loc, _data + _aligned_dim * from_loc, num_points * _aligned_dim * sizeof(data_t));
}

template <typename data_t> location_t InMemOOCDataStore<data_t>::calculate_medoid() const
{
    // allocate and init centroid
    float *center = new float[_aligned_dim];
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] = 0;

    for (size_t i = 0; i < this->capacity(); i++)
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] += (float)_data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= (float)this->capacity();

    // compute all to one distance
    float *distances = new float[this->capacity()];

    // TODO: REFACTOR. Removing pragma might make this slow. Must revisit.
    //  Problem is that we need to pass num_threads here, it is not clear
    //  if data store must be aware of threads!
    // #pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t)this->capacity(); i++)
    {
        // extract point and distance reference
        float &dist = distances[i];
        const data_t *cur_vec = _data + (i * (size_t)_aligned_dim);
        dist = 0;
        float diff = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            diff = (center[j] - (float)cur_vec[j]) * (center[j] - (float)cur_vec[j]);
            dist += diff;
        }
    }
    // find imin
    uint32_t min_idx = 0;
    float min_dist = distances[0];
    for (uint32_t i = 1; i < this->capacity(); i++)
    {
        if (distances[i] < min_dist)
        {
            min_idx = i;
            min_dist = distances[i];
        }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
}

template <typename data_t> Distance<data_t> *InMemOOCDataStore<data_t>::get_dist_fn() const
{
    return this->_distance_fn.get();
}

template DISKANN_DLLEXPORT class InMemOOCDataStore<float>;
template DISKANN_DLLEXPORT class InMemOOCDataStore<int8_t>;
template DISKANN_DLLEXPORT class InMemOOCDataStore<uint8_t>;

} // namespace diskann