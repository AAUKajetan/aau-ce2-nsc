# Miniproject 2 - Assignment Questions Summary

## Question 1: Parallel Version using Multiprocessing

### 1.1 Optimal Chunk Size for Different Number of Processes (P)

**Analysis Method:** Benchmarked all combinations of chunk sizes (32-1024) and process counts (1-10) across multiple resolutions (1024, 2048, 4096, 8192).

#### Key Findings

| Resolution | Optimal Chunk | Optimal P | Execution Time |
|------------|---------------|-----------|----------------|
| 1024×1024  | 256           | 6         | 0.775s         |
| 2048×2048  | 256           | 8         | 1.608s         |
| 4096×4096  | 256           | 10        | 4.648s         |
| 8192×8192  | 128           | 8         | 17.207s        |

**Conclusion:** The optimal chunk size remains relatively constant at **128-256 pixels** regardless of resolution. This is because:
- Too small chunks (≤64) → excessive inter-process communication (IPC) overhead
- Too large chunks (≥512) → poor load balancing (Mandelbrot boundary pixels take longer)
- The 128-256 range balances overhead vs. load distribution

---

### 1.2 Execution Time and Speed-up for Different P

#### 1024×1024 Resolution

| P (Processes) | Best Time (s) | Speedup vs P=1 |
|---------------|---------------|----------------|
| 1             | 1.59          | 1.0x           |
| 2             | 1.12          | 1.4x           |
| 4             | 0.95          | 1.7x           |
| **6**         | **0.78**      | **2.0x**       |
| 8             | 0.86          | 1.9x           |
| 10            | 0.96          | 1.7x           |

#### 2048×2048 Resolution

| P (Processes) | Best Time (s) | Speedup vs P=1 |
|---------------|---------------|----------------|
| 1             | 3.53          | 1.0x           |
| 2             | 2.37          | 1.5x           |
| 4             | 1.79          | 2.0x           |
| 6             | 1.61          | 2.2x           |
| **8**         | **1.61**      | **2.2x**       |
| 10            | 1.63          | 2.2x           |

#### Speed-up Analysis

```
Speedup vs Number of Processes (chunk=256, 2048×2048):

Speedup |
   2.5  |                    ●───●───●
        |              ●────●
   2.0  |        ●────●
        |
   1.5  |  ●────●
        |
   1.0  |●
        +──────────────────────────────
           1   2   4   6   8  10   P
```

**Observations:**
- Speedup scales sub-linearly (Amdahl's Law)
- Diminishing returns after P=6-8 due to:
  - Process creation/communication overhead
  - Memory bandwidth saturation
  - OS scheduler contention
- Optimal P ≈ **CPU_cores - 2** (leaves headroom for OS)

---

## Question 2: Dask Version

### 2.1 Execution Time for Multi-core Execution (Single Computer)

| Resolution | Dask chunk=128 | Dask chunk=256 | Dask chunk=512 | Best Time |
|------------|----------------|----------------|----------------|-----------|
| 256×256    | 0.035s         | 0.042s         | N/A            | 0.035s    |
| 512×512    | 0.154s         | 0.054s         | 0.160s         | 0.054s    |
| 1024×1024  | 0.567s         | 0.211s         | 0.186s         | 0.186s    |
| 2048×2048  | 2.151s         | 0.710s         | 0.444s         | 0.444s    |

**Optimal Dask Chunk Sizes:**
- Small resolutions (≤512): chunk = 128-256
- Large resolutions (≥1024): chunk = 512

---

### 2.2 Comparison: Dask vs NumPy Vectorized Implementation

| Resolution | NumPy (s) | Dask (s) | Speedup | Winner |
|------------|-----------|----------|---------|--------|
| 256×256    | 0.042     | 0.035    | 1.2x    | Dask   |
| 512×512    | 0.157     | 0.054    | 2.9x    | Dask   |
| 1024×1024  | 0.672     | 0.186    | 3.6x    | Dask   |
| 2048×2048  | 2.951     | 0.444    | 6.6x    | Dask   |

```
Speedup (Dask vs NumPy):

Speedup |
   7.0  |                          ●
   6.0  |
   5.0  |
   4.0  |                    ●
   3.0  |              ●
   2.0  |
   1.0  |        ●
        +──────────────────────────────
           256  512  1024  2048   Resolution
```

**Key Conclusions:**

1. **Dask outperforms NumPy at all resolutions** tested
2. **Speedup increases with resolution** - larger workloads benefit more from parallelization
3. **Chunk size matters significantly:**
   - 2048×2048 with chunk=64: 7.35s (slower than NumPy!)
   - 2048×2048 with chunk=512: 0.44s (6.6x faster than NumPy)
4. **Why Dask wins:**
   - Automatic parallelization across all CPU cores
   - Lazy evaluation reduces memory pressure
   - Efficient chunk-based computation

---

## Summary

| Question | Key Answer |
|----------|------------|
| Optimal chunk size | **128-256** (constant across resolutions) |
| Optimal P | **6-8** (CPU_cores - 2) |
| Speedup pattern | Sub-linear, diminishing returns after P=6-8 |
| Dask vs NumPy | **Dask 1.2-6.6x faster** (scales with resolution) |
| Best Dask chunk | **256-512** for large images |

**Final Recommendation:** Use **Dask with chunk_size=512** for production workloads - it provides the best performance with minimal code complexity.
