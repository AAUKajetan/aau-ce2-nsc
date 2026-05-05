# Miniproject 3 — Mandelbrot Set Benchmark

## Project Structure

```
miniproject3/
├── src/                      # Calculator implementations
│   ├── config.py             # MandelbrotConfig dataclass
│   ├── mb_calculator.py      # Abstract base class (MandelbrotCalculator)
│   ├── mb_native_calculator.py   # Pure Python nested-loop
│   ├── mb_numpy_calculator.py    # Vectorized NumPy
│   ├── mb_numba_calculator.py    # Numba JIT-compiled
│   ├── mb_multiprocess_calculator.py  # Multiprocessing + NumPy chunks
│   └── mb_cuda_calculator.py     # CUDA stub (not yet implemented)
├── benchmark_app/            # Benchmark application
│   ├── main.py               # Entry point
│   ├── benchmark_config.py   # BenchmarkConfig dataclass + calculator registry
│   ├── benchmark_runner.py   # BenchmarkRunner — times calculators
│   └── plot_maker.py         # PlotMaker — generates plots from results
├── test/                     # Tests
│   ├── calculator_test.py    # Generic tests for all calculators
│   └── test_helpers.py       # Unit tests for internal functions
└── README.md
```

## Components

### `src/` — Calculator Implementations

Each calculator inherits from `MandelbrotCalculator` and implements `calculate() -> np.ndarray`.

| Calculator | Description |
|---|---|
| `NativeCalculator` | Pure Python double loop. Slowest but simplest. |
| `NumpyCalculator` | Vectorized with NumPy masks. No extra dependencies. |
| `NumbaCalculator` | JIT-compiled with Numba. Near-C speed after warmup. |
| `MultiprocessCalculator` | Splits image into chunks, processes in parallel with `multiprocessing.Pool`. Uses `NumpyCalculator` per chunk. |
| `CudaCalculator` | Stub — not yet implemented. |

### `benchmark_app/` — Benchmark Application

- **`BenchmarkConfig`** — Configures what to benchmark: which calculators, resolution ramp-up, number of runs, warmup, and Mandelbrot set parameters.
- **`BenchmarkRunner`** — Executes the benchmark suite, returns a list of `BenchmarkResult` (mean/std/min times per calculator per resolution).
- **`PlotMaker`** — Generates plots from results:
  - `plot_time_vs_resolution()` — line plot with error bars
  - `plot_speedup(baseline)` — bar chart of speedup vs a baseline calculator
  - `plot_scaling()` — log-log plot of time vs total pixels

### `test/` — Tests

- `calculator_test.py` — generic contract tests (shape, dtype, bounds, determinism, cross-implementation agreement, 1x1 edge case)
- `test_helpers.py` — unit tests for `_generate_chunks`, `_process_chunk`, `_numba_compute`

## Requirements

```
numpy
numba
matplotlib
pytest
```

Install:
```bash
pip install numpy numba matplotlib pytest
```

## Running the Benchmark

```bash
cd miniproject3
python benchmark_app/main.py
```

Results (plots) are saved to `benchmark_app/results/` by default.

### Customising the Benchmark

Edit `benchmark_app/main.py` or create your own script:

```python
from benchmark_config import BenchmarkConfig
from benchmark_runner import BenchmarkRunner
from plot_maker import PlotMaker
from src.config import MandelbrotConfig

config = BenchmarkConfig(
    mb_config=MandelbrotConfig(max_iter=200, chunk_size=64),
    calculators=["numpy", "numba", "multiprocess"],
    resolutions=[128, 256, 512, 1024, 2048],
    num_runs=5,
    warmup_runs=2,
)

runner = BenchmarkRunner(config)
results = runner.run()

PlotMaker(results).make_all_plots(show=True)
```

## Running Tests

```bash
cd miniproject3
pytest test/ -v
```
