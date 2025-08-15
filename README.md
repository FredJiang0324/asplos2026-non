# PageANN: Page-based Approximate Nearest Neighbor Search

PageANN is a disk-based Approximate Nearest Neighbor Search (ANNS) framework that organizes computation and storage at **page granularity**. Logical page-nodes are directly mapped to physical SSD pages, with a co-designed **graph structure** and **on-disk layout**, plus lightweight in-memory routing and dynamic memory management.

This design shortens traversal paths, aligns I/O with SSD page size, reduces indexing overhead, and improves throughput under tight memory budgets.

> In experiments, PageANN achieves over **50% latency reduction** compared to state-of-the-art disk-based ANNS across diverse memory ratios, while maintaining comparable recall (see paper for details).

---

## Key Features
- **Page-based graph**: Logical nodes ≙ SSD pages, enabling fewer hops and I/O-aligned traversal.
- **Co-designed SSD layout**: Stores representative vectors and embedded inter-page topology to avoid extra reads.
- **Dynamic memory strategy**: Adapts cache/layout to varying memory budgets for faster search.
- **Scales to large datasets**: From millions to billions of vectors (e.g., SIFT1M/100M/1B).
- **Configurable PQ** and search knobs.

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
Use the following command to build an index. Replace <dataset_path> and <index_output_prefix> to fit your environment.

- If you select **SIFT100M**, set:
  --data_path data/sift100m/learn.100M.u8bin

SIFT1M example:
generate_page_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path data/sift1M/sift.1M.u8bin \
  --index_path_prefix data/sift1M/sift1m_index_R100_L100_PQ12_PageANN \
  --R 100 \
  --min_degree_per_node 70 \
  --num_PQ_chunks 12 \
  --memBudgetInGB 0.10 \
  --full_ooc false \
  --use_lsh false

General form (replace paths as needed):
generate_page_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path <dataset_path> \
  --index_path_prefix <index_output_prefix> \
  --R 100 \
  --min_degree_per_node 70 \
  --num_PQ_chunks 12 \
  --memBudgetInGB 2.0 \
  --full_ooc false \
  --use_lsh false

SIFT100M example:
generate_page_graph \
  --data_type uint8 \
  --dist_fn l2 \
  --data_path data/sift100m/learn.100M.u8bin \
  --index_path_prefix data/sift100m/sift100m_index_R100_L100_PQ12_PageANN \
  --R 100 \
  --min_degree_per_node 70 \
  --num_PQ_chunks 12 \
  --memBudgetInGB 2.0 \
  --full_ooc false \
  --use_lsh false

---

### 2) Compute Ground Truth
SIFT1M example:
compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file data/sift1M/sift.1M.u8bin \
  --query_file data/sift1M/query10K.u8bin \
  --origin_gt_file data/sift1M/query_1M_gt100 \
  --K 100 \
  --index_prefix data/sift1M/sift1m_index_R100_L100_PGD_PageANN \
  --gt_file data/sift1M/sift1M_gt_new_K100

SIFT100M example:
compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file data/sift100m/learn.100M.u8bin \
  --query_file data/sift100m/query.public.10K.u8bin \
  --origin_gt_file data/sift100m/gt_new_K100 \
  --K 100 \
  --index_prefix data/sift100m/ooc/sift100m_index_R110_L100_PQ12_PGD_PageANN \
  --gt_file data/sift100m/ooc/gt_new_K100

---

### 3) Search the PageANN Index
General form:
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
  --num_nodes_to_cache 0 \
  --use_lsh false \
  --use_subset_lsh false \
  --radius 0

SIFT1M example:
search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix data/sift1M/sift1m_index_R100_L100_PGD_PageANN \
  --pq_path_prefix data/sift1M/sift1m_index_R100_L100_PGD_PageANN_PQ12 \
  --query_file data/sift1M/query10K.u8bin \
  --gt_file data/sift1M/sift1M_gt_new_K100 \
  --result_path data/sift1M/res \
  -K 100 -L 100 -W 10 -T 15 \
  --num_nodes_to_cache 0 \
  --use_lsh false \
  --use_subset_lsh false \
  --radius 0

---

## Memory Budgets: Quick Recipes

PageANN adapts cache/layout to the given memory budget configured at **build time** via `--memBudgetInGB`. Tune search knobs (`-L`, `-W`, `-T`, `--num_nodes_to_cache`) per budget and hardware.

### A) Very Tight Memory (e.g., 0.5 GB)
Build:
... --memBudgetInGB 0.5 --num_PQ_chunks 12 --R 100 --min_degree_per_node 70 --full_ooc false --use_lsh false
Search:
- Use smaller `-W` (8–10) and moderate `-L` (80–100).
- Keep `--num_nodes_to_cache 0` or a small positive value if RAM allows.

### B) Moderate Memory (e.g., 2 GB)  [SIFT100M example path]
Build:
... --data_path data/sift100m/learn.100M.u8bin --memBudgetInGB 2.0 --num_PQ_chunks 12 --R 100 --min_degree_per_node 70 --full_ooc false --use_lsh false
Search:
- `-W 10–16`, `-L 100–150`, `-T 15–24` depending on CPU.
- Consider `--num_nodes_to_cache > 0` to reduce latency if RAM permits.

### C) Larger Memory (e.g., 8 GB+)
Build:
... --memBudgetInGB 8.0 --num_PQ_chunks 12 --R 100 --min_degree_per_node 70 --full_ooc false --use_lsh false
Search:
- Increase `--num_nodes_to_cache` for lower latency.
- Slightly higher `-W` can improve recall; adjust `-L` for your target recall/latency.

**General Guidance**
- `--memBudgetInGB` controls PageANN’s in-memory structures. Lower budgets reduce cache and can increase I/O; higher budgets allow more caching and typically reduce latency.
- Keep `--num_PQ_chunks` consistent between build and search artifacts.
- Tune `-L` (expansion) and `-W` (beamwidth) to hit your recall target (e.g., R@10=0.9) with acceptable latency.
- `-T` scales throughput; increase until CPU saturates or tail latency worsens.

---

## PQ Reordering for a New Budget
### This generates new PQ data to match a changed memory budget.
### Parameters: <program> <data_type> <base_data_path> <pq_output_prefix> <target_num_chunks>
generate_reorder_pq \
  uint8 \
  data/sift100m/learn.100M.u8bin \
  data/sift100m/sift100m_index_R110_L100_PQ12_PGD_PageANN \
  29

---

## Notes
- Some CLI flags are kept for binary compatibility; defaults above align with the design choices in the paper.

---

