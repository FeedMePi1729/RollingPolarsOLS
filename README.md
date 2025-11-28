# Polars Rolling OLS

A Polars-native wrapper around statsmodels' `RollingOLS` that provides seamless integration with Polars DataFrames for time series regression analysis.

## Motivation

Time series regression is fundamental to quantitative finance, econometrics, and data science. While Python offers excellent tools like statsmodels' `RollingOLS`, they primarily work with NumPy arrays and Pandas DataFrames. This creates friction for users who have adopted Polars for its superior performance and ergonomics.

**The problem this package solves:**

1. **Type conversion overhead**: Converting between Polars → Pandas/NumPy → back to Polars is tedious and error-prone
2. **Boilerplate reduction**: Eliminates repetitive code for extracting coefficients, t-values, fitted values, and residuals
3. **Polars-first design**: Works directly with Polars DataFrames using column names, not positional arrays
4. **Unified output**: Combines all regression statistics into a single, named DataFrame for easy analysis

## Features

- ✅ **Polars DataFrame input/output**: No manual conversions needed
- ✅ **Automatic column naming**: Coefficients and t-statistics named after your variables
- ✅ **Null handling**: Built-in strategies (`'drop'`, `'fill'`, `'ignore'`)
- ✅ **Lazy computation**: Results are cached to avoid redundant calculations
- ✅ **Comprehensive output**: Extract fitted values, residuals, parameters, t-values, or everything at once

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import polars as pl
from polars_rolling_ols import PolarsRollingOLS

# Create sample data
data = pl.DataFrame({
    'timestamp': pl.date_range(start='2020-01-01', end='2020-12-31', interval='1d'),
    'returns': [...],  # Your dependent variable
    'factor1': [...],  # Independent variable 1
    'factor2': [...],  # Independent variable 2
})

# Initialize rolling OLS with 30-day window
model = PolarsRollingOLS(
    data=data,
    endog='returns',
    exog=['factor1', 'factor2'],
    window=30,
    null_behavior='drop'
)

# Get all statistics in one DataFrame
stats = model.get_timeseries_stats(data['timestamp'])

# Or extract components individually
coefficients = model.get_params()      # factor1_coef, factor2_coef
tvalues = model.get_tvalues()          # factor1_tval, factor2_tval
fitted = model.get_fitted_values()     # fitted values
residuals = model.get_residuals()      # residuals
```

## API Reference

### `PolarsRollingOLS`

**Parameters:**
- `data` (pl.DataFrame): Input DataFrame
- `endog` (str): Column name for dependent variable
- `exog` (list[str]): List of column names for independent variables
- `window` (int): Rolling window size
- `null_behavior` (str): How to handle null values (`'drop'`, `'fill'`, `'ignore'`)
- `**kwargs`: Additional arguments passed to statsmodels' `RollingOLS`

**Methods:**
- `get_params()`: Returns DataFrame with rolling coefficients (`<var>_coef` columns)
- `get_tvalues()`: Returns DataFrame with t-statistics (`<var>_tval` columns)
- `get_fitted_values()`: Returns DataFrame with fitted values
- `get_residuals()`: Returns DataFrame with residuals
- `get_timeseries_stats(time_index)`: Returns combined DataFrame with all statistics
- `clear_cache()`: Clears cached results to free memory

## Use Cases

**Factor modeling**:
```python
# Rolling factor exposures
model = PolarsRollingOLS(
    data=returns_data,
    endog='portfolio_returns',
    exog=['market', 'smb', 'hml', 'momentum'],
    window=252  # 1-year rolling window
)
exposures = model.get_params()
```

**Time-varying betas**:
```python
# Calculate rolling beta to market
model = PolarsRollingOLS(
    data=stock_data,
    endog='stock_returns',
    exog=['market_returns'],
    window=60
)
betas = model.get_params()
```

**Pairs trading**:
```python
# Rolling hedge ratios
model = PolarsRollingOLS(
    data=pairs_data,
    endog='stock_a',
    exog=['stock_b'],
    window=20
)
hedge_ratios = model.get_params()
residuals = model.get_residuals()  # Mean-reverting spread
```

## Current Limitations & Improvement Opportunities

### Performance Bottlenecks

The current implementation inherits from statsmodels' `RollingOLS`, which has several performance limitations:

1. **NumPy-based computation**: All calculations happen in NumPy, requiring conversion from Polars
2. **Single-threaded execution**: No parallelization across windows or features
3. **No GPU acceleration**: Pure CPU implementation
4. **Memory overhead**: Stores full regression results for all windows

### Potential Speedups

**🚀 Native Polars/Rust implementation** (10-100x faster)
- Rewrite core regression logic in Rust using Polars' expression API
- Leverage Polars' lazy evaluation and query optimization
- Utilize SIMD operations for matrix operations

**🚀 Parallel window processing** (2-8x faster)
- Process independent rolling windows in parallel
- Use Polars' native parallelism capabilities
- Especially beneficial for large datasets with many windows

**🚀 GPU acceleration** (10-1000x faster for large datasets)
- Implement using cuDF/RAPIDS for GPU-based rolling operations
- Batch matrix operations across all windows on GPU
- Ideal for high-frequency data or large cross-sections

**🚀 Incremental computation** (2-5x faster)
- Cache intermediate matrix calculations (X'X, X'y)
- Update incrementally as window slides (Woodbury matrix identity)
- Avoid full recalculation for each window

**🚀 Memory optimization**
- Stream results instead of storing all windows
- Lazy evaluation of statistics (only compute what's needed)
- Compressed storage for coefficient histories

### Implementation Roadmap

**Phase 1: Benchmarking** (Current)
- Profile existing implementation
- Establish performance baselines
- Identify primary bottlenecks

**Phase 2: Native Polars Integration**
- Implement as Polars plugin/extension
- Use Polars expressions for rolling operations
- Maintain backward compatibility

**Phase 3: Advanced Optimizations**
- Add GPU support (optional dependency)
- Implement incremental updates
- Optimize for common use cases (single predictor, etc.)

## Contributing

Performance improvements are especially welcome! If you have ideas for:
- Algorithmic optimizations
- Parallelization strategies  
- Rust/native implementations
- GPU acceleration

Please open an issue or PR.

## Benchmarks

*Coming soon: Comparative benchmarks vs Pandas/statsmodels baseline*

## License

MIT

## Acknowledgments

Built on top of the excellent [statsmodels](https://www.statsmodels.org/) and [Polars](https://pola.rs/) libraries.
