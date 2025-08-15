# PageANN: Page-based Approximate Nearest Neighbor Search

PageANN is a disk-based ANNS framework that organizes computation and storage at **page granularity**. It maps logical page-nodes directly to physical SSD pages, co-designs the **graph structure** and **on-disk layout**, and introduces lightweight in-memory routing with dynamic memory management. The design shortens traversal paths, aligns I/O with SSD page size, reduces indexing overhead, and improves throughput under tight memory budgets.

> In experiments, PageANN achieves >50% latency improvement over state-of-the-art disk-based ANNS across diverse memory ratios while maintaining comparable recall (details in paper).

---

## Key Features
- **Page-based graph**: logical nodes ≙ SSD pages; fewer hops, I/O-aligned traversal.
- **Co-designed SSD layout**: stores representative vectors and embedded inter-page topology to avoid extra reads.
- **Dynamic memory strategy**: adapts cache & layout to memory budgets; lightweight routing for faster search.
- **Scales to large datasets**: supports SIFT1M/100M/1B and similar.
- **Configurable PQ** and search knobs.



## Building

### Linux build: Install the following packages through apt-get
```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```

#### Install Intel MKL
**Ubuntu 20.04 or newer**
```bash
sudo apt install libmkl-full-dev
```

**Earlier versions of Ubuntu**
Install Intel MKL either by downloading the oneAPI MKL installer or using apt (we tested with build 2019.4-070 and 2022.1.2.146).

**OneAPI MKL Installer**
```bash
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146.sh
sudo sh l_BaseKit_p_2022.1.2.146.sh -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s
```

#### Build
```bash
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

## Dependencies

- CMake 3.10+
- C++17 compatible compiler
- Rust (for Rust components)
- Python 3.7+ (for Python bindings)

## Commands

### PageANN with LSH

#### For Sift1M (uint8)

```bash
generate_page_graph --data_type uint8 --dist_fn l2 --data_path data/sift1M/sift.1M.u8bin --index_path_prefix data/sift1M/sift1m_diskANN_index_R100_L100 --min_degree_per_node 70 --num_PQ_chunks 12 --R 100 --full_ooc false --use_lsh true --memBudgetInGB 0.10
```

#### For Sift100M

```bash
./generate_page_graph --data_type uint8 --dist_fn l2 --data_path data/sift100m/learn.100M.u8bin --index_path_prefix data/sift100m/sift100m_vamana_index_R100_L100_PQ12 --R 100 --min_degree_per_node 70 --num_PQ_chunks 12 --use_lsh true --memBudgetInGB 2.0
```

#### Ground Truth Computation

```bash
compute_groundtruth --data_type uint8 --dist_fn l2 --base_file data/sift1M/sift.1M.u8bin --query_file data/sift1M/query10K.u8bin --origin_gt_file data/sift1M/query_1M_gt100 --K 100 --index_prefix data/sift1M/sift1m_diskANN_index_R100_L100_PGD703_PageANN --gt_file data/sift1M/sift1M_gt_new_K100
```

#### Search with LSH

```bash
search_disk_index --data_type uint8 --dist_fn l2 --index_path_prefix data/sift1M/sift1m_diskANN_index_R100_L100_PGD703_PageANN --pq_path_prefix data/sift1M/sift1m_diskANN_index_R100_L100_PGD703_PageANN_PQ14 --query_file data/sift1M/query10K.u8bin --gt_file data/sift1M/sift1M_gt_new_K100 --result_path data/sift1M/res -K 100 -L 100 -W 10 -T 15 --num_nodes_to_cache 0 --use_lsh true --use_subset_lsh false --radius 0
```

#### Optional LSH Generation

```bash
./generate_lsh --data_type uint8 --graph_index_prefix data/sift100m/vamana_sift100M_N18_R24_L150_PQ20_PGD447_PageANN --data_file_to_use data/sift100m/learn.100M.u8bin --recompute true --target_num_sampled_nodes 0
```

#### PQ Reordering for Memory Budget

```bash
generate_reorder_pq uint8 data/sift100m/learn.100M.u8bin data/sift100m/vamana_sift100M_R110_L100_PQ12_PGD639_PageANN 29
```

### Standard DiskANN

#### Build Index

```bash
build_disk_index --data_type uint8 --dist_fn l2 --data_path data/sift1M/sift.1M.u8bin --index_path_prefix data/sift1M/sift1m_diskANN_index_R64_L100_PQ12 -R 64 -L100 -B 0.012 -M1
```

#### Search Index

```bash
search_disk_index --data_type uint8 --dist_fn l2 --index_path_prefix data/sift1M/sift1m_diskANN_index_R64_L100_PQ12 --query_file data/sift1M/query10K.u8bin --gt_file data/sift1M/sift1M_gt_K100 --result_path data/sift1M/res -K 100 -L 100 -W 10 -T 15
```



## License

This project is based on DiskANN and is licensed under the MIT License.

## Acknowledgments

This work builds upon the DiskANN framework developed by Microsoft Research. We gratefully acknowledge the original DiskANN authors for their foundational contributions to disk-based approximate nearest neighbor search.

**Original DiskANN Repository**: [Microsoft/DiskANN](https://github.com/microsoft/DiskANN)

**Original License**: MIT License (Copyright (c) Microsoft Corporation)

## Citation

If you use this code in your research, please cite the original DiskANN paper and our PageANN work.

## Contact

For questions regarding this implementation, please refer to the original DiskANN repository or contact the research team through appropriate academic channels.
