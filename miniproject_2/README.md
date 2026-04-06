# Miniproject 2 Outline

## 1. Introduction
- Overview of implementations covered: NumPy baseline( imported from MiniProject 1), Multiprocessing, Dask (single-node), Dask Distributed (cluster)

```
miniproject_2/
│
├── config.py                          # Shared constants (bounds, resolutions, paths)
│
├── miniproject_1.py                   # Core Mandelbrot implementations (reused by all)
│   ├── visualize()                    # Plots a Mandelbrot array with matplotlib
│   ├── export_to_csv()                # Saves results or timing data to CSV
│   └── numpy_implementation()         # Vectorised NumPy — main reference implementation
│
├── mp_implementation.py               # Multiprocessing implementation
│   ├── export_to_csv()                # Local CSV helper
│   ├── generate_chunks()              # Splits the grid into sub-regions with coordinate bounds
│   ├── process_chunk()                # Worker function — runs numpy_implementation on one chunk
│   └── compute_mandelbrot_parallel()  # Orchestrates mp.Pool over all chunks
│
├── benchmark_chunk_size.py            # Chunk size optimisation experiment
│   ├── benchmark_chunk_process_combinations()  # Sweeps chunk sizes × process counts
│   ├── save_results()                 # Saves benchmark tuples to CSV
│   └── plot_results()                 # Plots execution time heatmap per resolution
│
├── dask_implementation.py             # Dask single-node implementation
│   ├── _mandelbrot_chunk()            # Pure-NumPy kernel — applied per chunk via map_blocks
│   ├── dask_implementation()          # Builds Dask array, maps kernel, calls .compute()
│   ├── benchmark_dask_vs_numpy()      # Benchmarks Dask vs NumPy across resolutions/chunk sizes
│   ├── plot_comparison()              # Plots execution time and speedup comparison
│   └── save_results()                 # Saves benchmark results to CSV
│
├── dask_distributed_implementation.py # Dask Distributed cluster implementation
│   ├── _mandelbrot_chunk()            # NumPy kernel (defined locally to avoid import on workers)
│   ├── _compute_subregion()           # Builds and computes a subregion from scalar bounds only
│   ├── distributed_map_blocks()       # Builds Dask array from delayed blocks → client.compute()
│   ├── distributed_futures()          # Submits row slices as explicit Futures → client.submit()
│   ├── benchmark_distributed()        # Benchmarks both strategies vs NumPy on the cluster
│   ├── plot_results()                 # Plots execution time and speedup for cluster results
│   └── save_results()                 # Saves benchmark results to CSV
│
└── results/                           # All benchmark outputs
    ├── chunk_benchmark_*.csv/png      # Chunk size × process sweep per resolution
    ├── dask_vs_numpy_benchmark.csv    # Dask single-node vs NumPy timings
    ├── dask_vs_numpy_comparison.png   # Visualisation of Dask single-node vs NumPy
    ├── distributed_benchmark.csv      # Cluster benchmark results
    ├── distributed_benchmark.png      # Visualisation of cluster results
    ├── multi_process_mb_set_*.csv     # Mandelbrot output arrays (MP runs)
    └── timing_results.csv             # General timing log for the basic mp implementation
```
---

## 2. Implementation Overview
- **NumPy baseline** — imported from the miniproject_1
- **Multiprocessing** — chunk-based workload split across P processes using `mp.Pool`
- **Dask (single-node)** — `da.map_blocks` distributing chunks across local threads/processes
- **Dask Distributed (cluster)** — `da.from_delayed` + `client.submit` across 4 worker VMs


---

## 3. Multiprocessing Results

### 3.1 Optimal Chunk Size Analysis
- ![Chart](../results/chunk_benchmark_8192.png)
- Key finding: what chunk size minimises overhead vs load imbalance at each resolution

### 3.2 Speedup vs Number of Processes
- Plot: execution time and speedup vs P for fixed resolutions
- Compare to ideal linear speedup (Amdahl's law)
- Discuss overhead from process creation and inter-process communication

---

## 4. Dask Single-Node Results

### 4.1 Execution Time vs NumPy
- Plot: `dask_vs_numpy_comparison.png`
- Table: avg time and speedup per resolution and chunk size (from `dask_vs_numpy_benchmark.csv`)

### 4.2 Effect of Chunk Size
- How chunk size affects scheduling overhead vs parallelism
- Optimal chunk size observations per resolution

---

## 5. Dask Distributed (Cluster) Results

### 5.1 Cluster Setup
- Head node (scheduler) + 4 worker VMs
- Python/NumPy version alignment challenges encountered

### 5.2 Distributed-MapBlocks vs Distributed-Futures
- Plot: `distributed_benchmark.png`
- Table: avg time and speedup per strategy, resolution, chunk size (from `distributed_benchmark.csv`)
- Differences in task scheduling between the two strategies

### 5.3 Cluster vs Single-Node Dask
- Speedup from adding worker nodes
- Network communication overhead at small resolutions

---

## 6. Reflections