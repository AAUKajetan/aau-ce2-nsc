# Miniproject 2 — Codebase Overview

```
miniproject_2/
│
├── config.py                          # Shared constants (bounds, resolutions, paths)
│
├── miniproject_1.py                   # Core Mandelbrot implementations (reused by all)
│   ├── visualize()                    # Plots a Mandelbrot array with matplotlib
│   ├── export_to_csv()                # Saves results or timing data to CSV
│   ├── native_python_implementation() # Pure Python nested loop (baseline, slow)
│   ├── numpy_implementation()         # Vectorised NumPy — main reference implementation
│   └── numba_implementation()         # Numba JIT version (currently disabled)
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
