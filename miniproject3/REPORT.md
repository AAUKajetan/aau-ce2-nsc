# Miniproject 3 — Report

## 1. Problem Description

The Mandelbrot set is defined as the set of complex numbers $c$ for which the iteration $z_{n+1} = z_n^2 + c$ (starting from $z_0 = 0$) does not diverge. For each pixel in a 2D image, we map its coordinates to a complex number $c$ and iterate until either $|z| > 2$ (escape) or a maximum iteration count is reached.

This is an **embarrassingly parallel** problem — each pixel is independent — making it ideal for exploring different parallelisation strategies: CPU vectorisation, JIT compilation, multiprocessing, and GPU computing.

The goal of this project is to implement the Mandelbrot computation using multiple approaches and compare their performance across increasing image resolutions.

## 2. Implementations

### 2.1 Pure Python (NativeCalculator)

[TODO: Brief description of the nested-loop approach. Mention it serves as the correctness baseline and why it's slow (Python interpreter overhead per pixel).]

### 2.2 NumPy (NumpyCalculator)

[TODO: Describe the vectorised approach using masked arrays. Explain how the entire grid is processed per iteration step, and why this is faster than pure Python but still limited by per-iteration kernel launches and intermediate array allocation.]

### 2.3 Numba JIT (NumbaCalculator)

[TODO: Describe using @jit(nopython=True) to compile the per-pixel loop to native code via LLVM. Explain why this approaches C-level speed for scalar loop-heavy code.]

### 2.4 Multiprocessing (MultiprocessCalculator)

[TODO: Describe the chunking strategy — splitting the image into tiles and distributing across a process pool. Mention the configurable base calculator (native/numpy/numba) and the trade-off between pool overhead and parallel throughput.]

### 2.5 CuPy (CupyCalculator)

[TODO: Describe the GPU-accelerated NumPy approach. Same masked-array algorithm as NumPy but executed on GPU memory via CuPy. Discuss why it's faster than CPU NumPy but slower than a fused kernel due to per-iteration kernel launches.]

### 2.6 Dask (DaskCalculator)

[TODO: Describe using dask.array.map_blocks with a NumPy kernel per chunk. Explain the synchronous scheduler limitation and why it doesn't provide parallelism in this configuration.]

### 2.7 CUDA @cuda.jit (CudaCalculator)

[TODO: Describe the explicit CUDA kernel using Numba's @cuda.jit. Each thread computes one pixel. Discuss the block size choice (16×16), the warp-size-multiple rule, and why fusing the iteration loop into a single kernel eliminates the overhead seen in CuPy.]

### 2.8 Software Design

All implementations inherit from an abstract base class `MandelbrotCalculator` with a single `calculate() -> np.ndarray` method. This polymorphic design enables:
- A unified benchmark runner that works with any implementation
- Parametrised tests that verify correctness across all calculators
- A registry pattern for configuration-driven benchmarking

## 3. Benchmarking Methodology

### 3.1 Hardware

| Component | Specification |
|---|---|
| CPU | [TODO: CPU model from device_scanner] |
| Cores | 64 logical (62 used for multiprocessing) |
| RAM | [TODO: GB] |
| GPU | [TODO: GPU model] |
| VRAM | [TODO: GB] |
| CUDA | Runtime 12.9 (Driver supports 13.0) |

### 3.2 Configuration

```python
calculators = ["numpy", "numba", "cupy", "cuda", "multiprocess_numpy", "multiprocess_numba"]
resolutions = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
max_iter = 100
chunk_size = 256
num_runs = 3
warmup_runs = 1
block_size = (16, 16)  # CUDA kernel
breakout_time = 5.5s   # Skip higher resolutions if exceeded
```

### 3.3 Timing Methodology

- Each calculator is warmed up at 64×64 to trigger JIT compilation (Numba, CUDA)
- Timing uses `time.perf_counter()` around the full `calculate()` call
- For CUDA: `copy_to_host()` implicitly synchronises the GPU before the timer stops
- 3 timed runs per (calculator, resolution) pair; mean and std are reported
- Total wall-clock time is measured (including device allocation and host←device transfer for GPU implementations)

### 3.4 Breakout Strategy

If a calculator's mean time exceeds 5.5s at a given resolution, higher resolutions are skipped. This keeps total benchmark time manageable while still capturing the scaling behaviour.

## 4. Experimental Results

### 4.1 Execution Times

| Resolution | NumPy | Numba | CuPy | CUDA | MP+NumPy | MP+Numba |
|---|---|---|---|---|---|---|
| 256×256 | 0.020s | 0.011s | 0.027s | 0.000s | 0.248s | 0.875s |
| 512×512 | 0.079s | 0.041s | 0.028s | 0.001s | 0.256s | 0.342s |
| 1024×1024 | 0.301s | 0.158s | 0.039s | 0.003s | 0.277s | 0.385s |
| 2048×2048 | 1.393s | 0.638s | 0.082s | 0.003s | 0.343s | 0.416s |
| 4096×4096 | 6.957s | 2.540s | 0.351s | 0.014s | 0.561s | 0.560s |
| 8192×8192 | — | 9.997s | 1.425s | 0.055s | 1.479s | 1.309s |
| 16384×16384 | — | — | 5.714s | 0.199s | 5.187s | 4.693s |
| 32768×32768 | — | — | — | 0.798s | 24.335s | 20.515s |
| 65536×65536 | — | — | — | 3.280s | — | — |

### 4.2 Speedup vs Numba

| Resolution | CUDA | CuPy | MP+NumPy | MP+Numba |
|---|---|---|---|---|
| 1024×1024 | 52.9× | 4.1× | 0.6× | 0.4× |
| 4096×4096 | 181.4× | 7.2× | 4.5× | 4.5× |
| 8192×8192 | 181.5× | 7.0× | 6.8× | 7.6× |

### 4.3 Plots

<p align="center">
  <img src="results/time_vs_resolution.png" width="700" alt="Time vs Resolution">
</p>

<p align="center">
  <img src="results/speedup.png" width="700" alt="Speedup">
</p>

## 5. Analysis and Interpretation

### 5.1 CUDA @cuda.jit Dominance

[TODO: Explain why the fused CUDA kernel achieves 181× speedup over Numba. Discuss single kernel launch, register-only computation, and pixel-level parallelism.]

### 5.2 CuPy vs @cuda.jit

[TODO: Explain the 28× gap between CuPy and @cuda.jit at 4096. Discuss per-iteration kernel launches, global memory traffic for intermediates, and mask overhead.]

### 5.3 CPU Scaling

[TODO: Compare Numba (single-threaded JIT) vs NumPy (vectorised) vs Multiprocessing. Discuss where multiprocessing overhead pays off and where it doesn't.]

### 5.4 Block Size Analysis (Warp-Size-Multiple Rule)

The default block size of 16×16 = 256 threads was chosen because:
- 256 = 8 × 32 warps — zero wasted execution slots
- 2D mapping fits the image grid naturally
- Multiple blocks can co-reside on an SM, maximising occupancy

Block sizes that are NOT multiples of 32 (e.g., 20×20 = 400 threads) leave partial warps with idle threads, reducing throughput.

[TODO: If you ran experiments with different block sizes, include results here.]

### 5.5 Data Transfer Overhead

[TODO: Discuss that the benchmark includes device allocation + copy_to_host(). Note that no host→device transfer is needed (coordinates computed from scalar kernel arguments). At 16384×16384, the ~1GB int32 transfer takes ~40ms over PCIe Gen4, negligible vs 0.2s kernel time.]

### 5.6 Warp Divergence

[TODO: Discuss how pixels near the Mandelbrot boundary cause threads within a warp to diverge (some escape early, others iterate to max_iter). The warp serialises divergent branches. Despite this, GPU parallelism still dominates.]

### 5.7 Memory Type Considerations

| Variable | Memory Type | Rationale |
|---|---|---|
| xmin, xmax, ymin, ymax | Registers (kernel args) | Scalar constants |
| z_real, z_imag, c_real, c_imag | Registers (local) | Per-thread private state |
| output[row, col] | Global memory | Result array for host transfer |

The Mandelbrot kernel needs no shared memory because there is no data sharing between threads — each pixel is fully independent. Shared memory would benefit a **reduction** (e.g., computing mean iteration count) where block-local sums are accumulated before a global combine.

### 5.8 ARM Results (Apple M2 Pro)

[TODO: Brief comparison with ARM results if applicable. Discuss differences in GPU availability (Metal vs CUDA), CPU core count, and how the multiprocessing results compare.]

## 6. Conclusion

[TODO: Summarise key findings — CUDA @cuda.jit is the optimal approach for this embarrassingly parallel problem, achieving 181× speedup over single-threaded JIT. Multiprocessing is effective but limited by IPC overhead and core count. CuPy offers a convenient GPU path but cannot match a fused kernel. The abstract base class design made it straightforward to add and benchmark new implementations.]

## 7. References

- Numba CUDA documentation: https://numba.readthedocs.io/en/stable/cuda/index.html
- CuPy documentation: https://docs.cupy.dev/en/stable/
- NVIDIA CUDA Programming Guide (warp execution model): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
