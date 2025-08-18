# PageANN: Page-based Approximate Nearest Neighbor Search

This repository provides the artifacts for PageANN evaluation.  
We provide a ready-to-use **1M dataset index and ground-truth file** so that reviewers can quickly run the search command.  
For larger datasets such as **SIFT100M**, the same commands apply with path changes.

---

## 1. Search (Quick Start)

Reviewers may start directly from a provided **1M index and ground-truth** file.

General form
search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix <index_output_prefix> \
  --pq_path_prefix <pq_prefix> \
  --query_file <query_vectors> \
  --gt_file <groundtruth_file> \
  --result_path <search_results> \
  -K 100 -L 100 -W 10 -T 15 \
  --num_nodes_to_cache 0 \
  --use_lsh false \
  --use_subset_lsh false \
  --radius 0

SIFT100M example
search_disk_index \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix data/sift100m/sift100m_index_R100_L100_PGD_PageANN \
  --pq_path_prefix data/sift100m/sift100m_index_R100_L100_PGD_PageANN_PQ12 \
  --query_file data/sift100m/query.public.10K.u8bin \
  --gt_file data/sift100m/gt_new_K100 \
  --result_path data/sift100m/res \
  -K 100 -L 100 -W 10 -T 15 \
  --num_nodes_to_cache 0 \
  --use_lsh false \
  --use_subset_lsh false \
  --radius 0

---

## 2. Pre-processing

If reviewers wish to reproduce the index and ground-truth, the following commands apply.

### a) Build a PageANN Index
General form
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

SIFT100M example
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

### b) Compute Ground Truth
General form
compute_groundtruth \
  --data_type uint8 \
  --dist_fn l2 \
  --base_file <base_vectors> \
  --query_file <query_vectors> \
  --origin_gt_file <original_gt> \
  --K 100 \
  --index_prefix <index_prefix> \
  --gt_file <output_gt>

SIFT100M example
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

## Notes
- Provided 1M index and ground-truth allow reviewers to directly test search without full preprocessing.  
- For larger datasets, replace the paths accordingly.  
- Some CLI flags are kept for binary compatibility; defaults above match the paper settings.
