from statsmodels.regression.rolling import RollingOLS
import polars as pl
import warnings

warnings.filterwarnings("ignore")

class PolarsRollingOLS(RollingOLS):
    """Wrapper around statsmodels RollingOLS with Polars-friendly outputs.
    
    Caches fit results and provides methods to extract predictions, residuals,
    coefficients, and t-values as Polars DataFrames with proper column naming.
    
    Parameters
    ----------
    endog : array-like
        Dependent variable (1-D array or Series)
    exog : array-like or DataFrame
        Independent variables. If DataFrame, uses column names for coefficients.
    window : int
        Rolling window size
    coef_names : list[str], optional
        Names for coefficient columns. If None and exog is not a DataFrame,
        uses ['var_0', 'var_1', ...]. Ignored if exog is a DataFrame.
    **kwargs
        Additional arguments passed to RollingOLS
    """
    
    def __init__(
        self,
        data: pl.DataFrame, 
        endog: str, # dependent variable
        exog: list[str], # independent variable
        window: int,
        null_behavior: str = 'drop', # how to handle NaNs, allowed: 'drop', 'ignore', 'fill' 
        **kwargs
    ):
        
        # Store metadata
        self._window = window
        self._fit_result = None
        self._fitted_values = None
        self._fitted_values_cache = None
        self._residuals_cache = None
        
        # store data for later use
        
        if null_behavior == 'drop':
            data = data.drop_nulls(subset=[endog, *exog])
        
        elif null_behavior == 'fill':
            data = data.fill_null(strategy='forward').fill_null(strategy='backward')
            
        self._data = data
        self._endog = endog
        self._exog = exog
        self._endog_data = data.select(endog).to_numpy().flatten() # ensure 1-D
        self._exog_data = data.select(exog).to_numpy()
        
        
        super().__init__(self._endog_data, self._exog_data, window, **kwargs)
        
        # coefficient names
        self._coef_names = exog
        
    
    @property
    def fit_result(self):
        """Lazy-load fit result to avoid duplicate computation."""
        if self._fit_result is None:
            self._fit_result = self.fit()
        return self._fit_result
    
    @property
    def fitted_values(self):
        """Lazy-load fitted values to avoid duplicate computation."""
        
        if self._fitted_values is None:
            self._fitted_values = (self._exog_data * self.fit_result.params).sum(axis=1)
        return self._fitted_values
    
    def get_fitted_values(self) -> pl.DataFrame:
        """Extract fitted values as Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
            Single-column DataFrame with fitted values (predictions)
        """
        if self._fitted_values_cache is None:
            # Use statsmodels' built-in predictions
            fitted = self.fitted_values
            self._fitted_values_cache = pl.DataFrame({'fitted': fitted})
        return self._fitted_values_cache
    
    def get_residuals(self) -> pl.DataFrame:
        """Extract residuals as Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
            Single-column DataFrame with residuals (observed - fitted)
        """
        if self._residuals_cache is None:
            # Use actual residuals, not MSE
            resids = self._endog_data - (self.fit_result.params * self._exog_data).sum(axis=1) # type: ignore
            self._residuals_cache = pl.DataFrame({'residuals': resids})
        return self._residuals_cache
    
    def get_params(self) -> pl.DataFrame:
        """Extract rolling coefficients as Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with coefficient columns named '<var>_coef'
        """
        params = self.fit_result.params
        
        # Rename columns using stored coefficient names
        rename_map = {
            f'column_{i}': f'{name}_coef' # dependent on statsmodel naming columns as column_0, column_1, ...
            for i, name in enumerate(self._coef_names)
        }
        
        return pl.DataFrame(params).rename(rename_map)
    
    def get_tvalues(self) -> pl.DataFrame:
        """Extract rolling t-statistics as Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with t-stat columns named '<var>_tval'
        """
        tvals = self.fit_result.tvalues
        
        # Rename columns using stored coefficient names
        rename_map = {
            f'column_{i}': f'{name}_tval' 
            for i, name in enumerate(self._coef_names)
        }
        
        return pl.DataFrame(tvals).rename(rename_map)
    
    def get_timeseries_stats(self, time_index: pl.DataFrame | pl.Series) -> pl.DataFrame:
        """Combine all regression outputs into single DataFrame.
        
        Parameters
        ----------
        time_index : pl.Series
            Time index series to join with (must match length of results)
        
        Returns
        -------
        pl.DataFrame
            Combined DataFrame with time index, fitted values, residuals,
            coefficients, and t-values
        """
        # Build list of DataFrames to concatenate
        
        time = time_index if isinstance(time_index, pl.DataFrame) else pl.DataFrame({time_index.name or 'timestamp': time_index})
        
        dfs = [
            time,
            self._data.select(self._endog),
            self._data.select(self._exog),
            self.get_fitted_values(),
            self.get_residuals(),
            self.get_params(),
            self.get_tvalues(),
        ]
        
        # Horizontal concat (all same length)
        return pl.concat(dfs, how='horizontal')
    
    def clear_cache(self):
        """Clear cached results to free memory or force recomputation."""
        self._fit_result = None
        self._fitted_values_cache = None
        self._residuals_cache = None
        self._fitted_values = None