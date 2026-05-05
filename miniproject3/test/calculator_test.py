import numpy as np
import pytest

from src import (
    MandelbrotConfig,
    NativeCalculator,
    NumpyCalculator,
    NumbaCalculator,
    MultiprocessCalculator,

)

# Small config for fast tests
SMALL_CONFIG = MandelbrotConfig(
    xmin=-2.0, xmax=1.0,
    ymin=-1.5, ymax=1.5,
    width=64, height=64,
    max_iter=50,
    chunk_size=32,
)

# All calculators that should pass the generic tests
CALCULATORS = [
    NativeCalculator,
    NumpyCalculator,
    NumbaCalculator,
    MultiprocessCalculator,
]


@pytest.fixture(params=CALCULATORS, ids=lambda cls: cls.__name__)
def calculator(request):
    """Parametrized fixture: yields one calculator instance per implementation."""
    return request.param(SMALL_CONFIG)


# ─── Generic tests that every calculator must pass ───────────────────────────


class TestCalculatorContract:
    """Tests that apply to ALL calculator implementations."""

    def test_output_shape(self, calculator):
        result = calculator.calculate()
        assert result.shape == (SMALL_CONFIG.height, SMALL_CONFIG.width)

    def test_output_dtype_is_integer(self, calculator):
        result = calculator.calculate()
        assert np.issubdtype(result.dtype, np.integer)

    def test_iteration_bounds(self, calculator):
        """All values should be between 1 and max_iter (inclusive)."""
        result = calculator.calculate()
        assert result.min() >= 1
        assert result.max() <= SMALL_CONFIG.max_iter

    def test_center_point_stays_in_set(self, calculator):
        """Origin (0+0j) is in the Mandelbrot set, should reach max_iter."""
        result = calculator.calculate()
        # Find pixel closest to (0, 0)
        col = int((0 - SMALL_CONFIG.xmin) / (SMALL_CONFIG.xmax - SMALL_CONFIG.xmin) * (SMALL_CONFIG.width - 1))
        row = int((0 - SMALL_CONFIG.ymin) / (SMALL_CONFIG.ymax - SMALL_CONFIG.ymin) * (SMALL_CONFIG.height - 1))
        assert result[row, col] == SMALL_CONFIG.max_iter

    def test_corner_escapes_quickly(self, calculator):
        """Point far from set (e.g. top-right corner) should escape fast."""
        result = calculator.calculate()
        # Top-right corner corresponds to (xmax, ymax) which is (1, 1.5)
        # |c| > 2 so it escapes in 1 iteration
        assert result[-1, -1] < SMALL_CONFIG.max_iter

    def test_deterministic(self, calculator):
        """Two runs with same config produce identical results."""
        r1 = calculator.calculate()
        r2 = calculator.calculate()
        np.testing.assert_array_equal(r1, r2)

    def test_all_implementations_agree(self):
        """All calculators produce the same result for the same config."""
        results = [cls(SMALL_CONFIG).calculate() for cls in CALCULATORS]
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg=f"{CALCULATORS[0].__name__} != {CALCULATORS[i].__name__}"
            )

    def test_1x1_grid_no_crash(self):
        """A 1x1 grid should not crash and return correct shape."""
        cfg_1x1 = MandelbrotConfig(width=1, height=1, max_iter=50)
        for cls in CALCULATORS:
            result = cls(cfg_1x1).calculate()
            assert result.shape == (1, 1)
            assert result[0, 0] >= 1
