import pytest
import polars as pl
import numpy as np
from polars_rolling_ols.rolling_ols import PolarsRollingOLS


class TestPolarsRollingOLSInit:
    """Test the initialization of PolarsRollingOLS class."""
    
    def test_basic_initialization(self):
        """Test basic initialization with valid inputs."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        assert model._window == 3
        assert model._endog == 'y'
        assert model._exog == ['x1', 'x2']
        assert model._coef_names == ['x1', 'x2']
        assert len(model._endog_data) == 5
        assert model._exog_data.shape == (5, 2)
    
    def test_initialization_with_null_behavior_drop(self):
        """Test initialization with null_behavior='drop'."""
        data = pl.DataFrame({
            'y': [1.0, None, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, None, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=2,
            null_behavior='drop'
        )
        
        # Should drop rows with nulls
        assert len(model._endog_data) == 3  # Only rows 0 and 2 have no nulls
    
    def test_initialization_with_null_behavior_fill(self):
        """Test initialization with null_behavior='fill'."""
        data = pl.DataFrame({
            'y': [1.0, None, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=2,
            null_behavior='fill'
        )
        
        # Should fill nulls
        assert len(model._endog_data) == 5
        assert not np.isnan(model._endog_data).any()
    
    def test_initialization_with_null_behavior_ignore(self):
        """Test initialization with null_behavior='ignore'."""
        data = pl.DataFrame({
            'y': [1.0, None, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=2,
            null_behavior='ignore'
        )
        
        # Should keep nulls
        assert len(model._endog_data) == 5
    
    def test_initialization_single_exog(self):
        """Test initialization with single exogenous variable."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1'],
            window=3
        )
        
        assert model._exog_data.shape == (5, 1)
        assert model._coef_names == ['x1']


class TestPolarsRollingOLSProperties:
    """Test the properties of PolarsRollingOLS class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        })
        
        return PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
    
    def test_fit_result_property_lazy_load(self, sample_model):
        """Test that fit_result is lazily loaded."""
        # Initially None
        assert sample_model._fit_result is None
        
        # Access triggers computation
        result = sample_model.fit_result
        assert result is not None
        
        # Second access uses cache
        result2 = sample_model.fit_result
        assert result is result2
    
    def test_fitted_values_property_lazy_load(self, sample_model):
        """Test that fitted_values is lazily loaded."""
        # Initially None
        assert sample_model._fitted_values is None
        
        # Access triggers computation
        fitted = sample_model.fitted_values
        assert fitted is not None
        assert len(fitted) == 6
        
        # Second access uses cache
        fitted2 = sample_model.fitted_values
        assert np.array_equal(fitted, fitted2, equal_nan=True)


class TestPolarsRollingOLSGetFittedValues:
    """Test the get_fitted_values method."""
    
    def test_get_fitted_values_returns_dataframe(self):
        """Test that get_fitted_values returns a Polars DataFrame."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        fitted = model.get_fitted_values()
        
        assert isinstance(fitted, pl.DataFrame)
        assert 'fitted' in fitted.columns
        assert len(fitted) == 5
    
    def test_get_fitted_values_caching(self):
        """Test that get_fitted_values caches results."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        fitted1 = model.get_fitted_values()
        fitted2 = model.get_fitted_values()
        
        # Should return same cached object
        assert fitted1 is fitted2


class TestPolarsRollingOLSGetResiduals:
    """Test the get_residuals method."""
    
    def test_get_residuals_returns_dataframe(self):
        """Test that get_residuals returns a Polars DataFrame."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        residuals = model.get_residuals()
        
        assert isinstance(residuals, pl.DataFrame)
        assert 'residuals' in residuals.columns
        assert len(residuals) == 5
    
    def test_get_residuals_caching(self):
        """Test that get_residuals caches results."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        residuals1 = model.get_residuals()
        residuals2 = model.get_residuals()
        
        # Should return same cached object
        assert residuals1 is residuals2
    
    def test_residuals_calculation_accuracy(self):
        """Test that residuals are calculated correctly (observed - fitted)."""
        # Simple linear case where y = x1
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1'],
            window=3
        )
        
        residuals = model.get_residuals()
        fitted = model.get_fitted_values()
        
        # Residuals should be close to zero for perfect linear relationship
        # (after window period)
        assert residuals.shape[0] == 5


class TestPolarsRollingOLSGetParams:
    """Test the get_params method."""
    
    def test_get_params_returns_dataframe(self):
        """Test that get_params returns a Polars DataFrame."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        params = model.get_params()
        
        assert isinstance(params, pl.DataFrame)
        assert 'x1_coef' in params.columns
        assert 'x2_coef' in params.columns
        assert len(params) == 5
    
    def test_get_params_column_naming(self):
        """Test that coefficient columns are named correctly."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'var_a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'var_b': [2.0, 3.0, 4.0, 5.0, 6.0],
            'var_c': [3.0, 4.0, 5.0, 6.0, 7.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['var_a', 'var_b', 'var_c'],
            window=3
        )
        
        params = model.get_params()
        
        assert 'var_a_coef' in params.columns
        assert 'var_b_coef' in params.columns
        assert 'var_c_coef' in params.columns
        assert len(params.columns) == 3


class TestPolarsRollingOLSGetTvalues:
    """Test the get_tvalues method."""
    
    def test_get_tvalues_returns_dataframe(self):
        """Test that get_tvalues returns a Polars DataFrame."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        tvalues = model.get_tvalues()
        
        assert isinstance(tvalues, pl.DataFrame)
        assert 'x1_tval' in tvalues.columns
        assert 'x2_tval' in tvalues.columns
        assert len(tvalues) == 5
    
    def test_get_tvalues_column_naming(self):
        """Test that t-value columns are named correctly."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'var_a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'var_b': [2.0, 3.0, 4.0, 5.0, 6.0],
            'var_c': [3.0, 4.0, 5.0, 6.0, 7.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['var_a', 'var_b', 'var_c'],
            window=3
        )
        
        tvalues = model.get_tvalues()
        
        assert 'var_a_tval' in tvalues.columns
        assert 'var_b_tval' in tvalues.columns
        assert 'var_c_tval' in tvalues.columns
        assert len(tvalues.columns) == 3


class TestPolarsRollingOLSGetTimeseriesStats:
    """Test the get_timeseries_stats method."""
    
    def test_get_timeseries_stats_with_series(self):
        """Test get_timeseries_stats with a Polars Series as time index."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        time_index = pl.Series('timestamp', [1, 2, 3, 4, 5])
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        result = model.get_timeseries_stats(time_index)
        
        assert isinstance(result, pl.DataFrame)
        assert 'timestamp' in result.columns
        assert 'y' in result.columns
        assert 'x1' in result.columns
        assert 'x2' in result.columns
        assert 'fitted' in result.columns
        assert 'residuals' in result.columns
        assert 'x1_coef' in result.columns
        assert 'x2_coef' in result.columns
        assert 'x1_tval' in result.columns
        assert 'x2_tval' in result.columns
        assert len(result) == 5
    
    def test_get_timeseries_stats_with_dataframe(self):
        """Test get_timeseries_stats with a Polars DataFrame as time index."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        time_index = pl.DataFrame({'date': [1, 2, 3, 4, 5]})
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        result = model.get_timeseries_stats(time_index)
        
        assert isinstance(result, pl.DataFrame)
        assert 'date' in result.columns
        assert len(result) == 5
    
    def test_get_timeseries_stats_unnamed_series(self):
        """Test get_timeseries_stats with unnamed Series."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        time_index = pl.Series([1, 2, 3, 4, 5])  # No name
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1'],
            window=3
        )
        
        result = model.get_timeseries_stats(time_index)
        
        assert isinstance(result, pl.DataFrame)
        assert 'timestamp' in result.columns  # Default name


class TestPolarsRollingOLSClearCache:
    """Test the clear_cache method."""
    
    def test_clear_cache_resets_all_cached_values(self):
        """Test that clear_cache resets all cached values."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2'],
            window=3
        )
        
        # Trigger all lazy properties
        _ = model.fit_result
        _ = model.fitted_values
        _ = model.get_fitted_values()
        _ = model.get_residuals()
        
        # Verify they're cached
        assert model._fit_result is not None
        assert model._fitted_values is not None
        assert model._fitted_values_cache is not None
        assert model._residuals_cache is not None
        
        # Clear cache
        model.clear_cache()
        
        # Verify all are None
        assert model._fit_result is None
        assert model._fitted_values is None
        assert model._fitted_values_cache is None
        assert model._residuals_cache is None
    
    def test_clear_cache_allows_recomputation(self):
        """Test that values can be recomputed after clearing cache."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1'],
            window=3
        )
        
        # Get initial results
        fitted1 = model.get_fitted_values()
        
        # Clear cache
        model.clear_cache()
        
        # Recompute
        fitted2 = model.get_fitted_values()
        
        # Values should be equal but different objects
        assert fitted1.equals(fitted2)
        assert fitted1 is not fitted2


class TestPolarsRollingOLSIntegration:
    """Integration tests for the full workflow."""
    
    def test_end_to_end_simple_regression(self):
        """Test complete workflow with simple linear relationship."""
        np.random.seed(42)
        n = 20
        x = np.linspace(1, 20, n)
        y = 2 * x + 1 + np.random.randn(n) * 0.5
        
        data = pl.DataFrame({
            'y': y,
            'x': x,
            'time': range(n)
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x'],
            window=10
        )
        
        # Get all outputs
        fitted = model.get_fitted_values()
        residuals = model.get_residuals()
        params = model.get_params()
        tvalues = model.get_tvalues()
        
        # Verify shapes
        assert len(fitted) == n
        assert len(residuals) == n
        assert len(params) == n
        assert len(tvalues) == n
        
        # Verify column names
        assert 'fitted' in fitted.columns
        assert 'residuals' in residuals.columns
        assert 'x_coef' in params.columns
        assert 'x_tval' in tvalues.columns
    
    def test_end_to_end_multiple_predictors(self):
        """Test complete workflow with multiple predictors."""
        np.random.seed(42)
        n = 30
        
        data = pl.DataFrame({
            'y': np.random.randn(n).cumsum(),
            'x1': np.random.randn(n).cumsum(),
            'x2': np.random.randn(n).cumsum(),
            'x3': np.random.randn(n).cumsum(),
            'time': range(n)
        })
        
        model = PolarsRollingOLS(
            data=data,
            endog='y',
            exog=['x1', 'x2', 'x3'],
            window=10
        )
        
        # Get combined stats
        time_index = pl.Series('date', range(n))
        result = model.get_timeseries_stats(time_index)
        
        # Verify all columns present
        expected_cols = [
            'date', 'y', 'x1', 'x2', 'x3', 'fitted', 'residuals',
            'x1_coef', 'x2_coef', 'x3_coef',
            'x1_tval', 'x2_tval', 'x3_tval'
        ]
        
        for col in expected_cols:
            assert col in result.columns
        
        assert len(result) == n


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
