# PageANN: Page-based Approximate Nearest Neighbor Search

PageANN is a disk-based Approximate Nearest Neighbor Search (ANNS) framework that organizes computation and storage at **page granularity**. Logical page-nodes are directly mapped to physical SSD pages, with a co-designed **graph structure** and **on-disk layout**, along with lightweight in-memory routing and dynamic memory management.

This design shortens traversal paths, aligns I/O with SSD page size, reduces indexing overhead, and improves throughput under tight memory budgets.

> In experiments, PageANN achieves over **50% latency reduction** compared to state-of-the-art disk-based ANNS across diverse memory ratios, while maintaining comparable recall (see paper for details).

---

## Key Features
- **Page-based graph**: Logical nodes ≙ SSD pages, enabling fewer hops and I/O-aligned traversal.
- **Co-designed SSD layout**: Stores representative vectors and embedded inter-page topology to avoid extra reads.
- **Dynamic memory strategy**: Adapts cache and layout to varying memory budgets for faster search.
- **Large-scale scalability**: Supports datasets from millions to billions of vectors (e.g., SIFT1M/100M/1B).
- **Configurable PQ** and search parameters.

---

## Build Instructions

### 1) Install Dependencies (Linux)
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev

### 2) Install Intel MKL
# Ubuntu 20.04+
sudo apt install libmkl-full-dev

# Earlier versions (tested 2019.4-070 and 2022.1.2.146)
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146.sh
sudo sh l_BaseKit_p_2022.1.2.146.sh -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s

### 3) Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

### Requirements
- CMake ≥ 3.10
- C++17-compatible compiler
- Rust (for Rust components)
- Python ≥ 3.7 (for Python bindings)

---

## Example Usage

### 1) Build a PageANN Index
Use the following command to build an index. Replace <dataset_path> with the path to your dataset and <index_output_prefix> with your desired output prefix.

- If you select **SIFT100M**, set:
  --data_path data/sift100m/learn.100M.u8bin

Command:
generate_page_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path <dataset_path> \
  --index_path_prefix <index_output_prefix> \
  --R 100 \
  --min_degree_per_node 70 \
  --num_PQ_chunks 12 \
  --memBudgetInGB 2.0

#### Memory Budget (--memBudgetInGB)
- Example: --memBudgetInGB 0.5  → very tight budget; smaller in-memory cache; lower memory use but slower search.
- Example: --memBudgetInGB 4.0  → larger budget; more caching; typically faster queries.
Adjust based on your available RAM. PageANN dynamically adapts cache and layout under different budgets.

---

### 2) Compute Ground Truth
compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file <base_vectors> \
  --query_file <query_vectors> \
  --origin_gt_file <original_gt> \
  --K 100 \
  --gt_file <output_gt>

---

### 3) Search the PageANN Index
search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix <index_output_prefix> \
  --pq_path_prefix <pq_prefix> \
  --query_file <query_vectors> \
  --gt_file <groundtruth_file> \
  --result_path <search_results> \
  -K 100 \
  -L 100 \
  -W 10 \
  -T 15 \
  --num_nodes_to_cache 0

---