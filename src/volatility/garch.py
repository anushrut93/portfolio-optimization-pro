"""
GARCH Volatility Modeling and Forecasting Module.

This module implements GARCH (Generalized Autoregressive Conditional 
Heteroskedasticity) models for volatility forecasting in portfolio optimization.
It provides tools for fitting various GARCH specifications, generating volatility
forecasts, and integrating with portfolio construction.

Key Features:
- GARCH(1,1), GJR-GARCH, and EGARCH models
- Multi-step ahead volatility forecasting
- Risk-based portfolio weighting
- Volatility regime detection
- Integration with existing optimization framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from arch import arch_model
from arch.univariate import GARCH, EGARCH, ConstantMean, ZeroMean
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class GARCHResults:
    """
    Container for GARCH model results and forecasts.
    
    This class stores fitted model parameters, conditional volatility,
    and forecasts for use in portfolio optimization.
    """
    model_type: str
    parameters: Dict[str, float]
    conditional_volatility: pd.Series
    volatility_forecast: np.ndarray
    forecast_horizon: int
    aic: float
    bic: float
    log_likelihood: float
    
    def annualized_forecast(self, periods_per_year: int = 252) -> np.ndarray:
        """Convert daily volatility forecast to annualized."""
        return self.volatility_forecast * np.sqrt(periods_per_year)
    
    def __str__(self) -> str:
        """String representation of results."""
        return f"""
GARCH Model Results
==================
Type: {self.model_type}
Parameters: {self.parameters}
AIC: {self.aic:.2f}
BIC: {self.bic:.2f}
Forecast Horizon: {self.forecast_horizon} days
Current Volatility: {self.conditional_volatility.iloc[-1]:.4f}
"""


class GARCHModeler:
    """
    Main class for GARCH modeling and volatility forecasting.
    
    This class provides methods for fitting various GARCH specifications,
    generating forecasts, and creating volatility-based portfolio weights.
    """
    
    def __init__(self, returns_data: Union[pd.Series, pd.DataFrame]):
        """
        Initialize GARCH modeler with returns data.
        
        Args:
            returns_data: Series or DataFrame of returns (in decimal form)
        """
        self.returns_data = returns_data
        self.fitted_models = {}
        self.results = {}
        
    def fit_garch(
        self,
        series: pd.Series,
        model_type: str = 'GARCH',
        p: int = 1,
        q: int = 1,
        o: int = 0,
        dist: str = 't',
        mean: str = 'constant'
    ) -> GARCHResults:
        """Fit GARCH model to returns series."""
        # Ensure series has a name
        if series.name is None:
            series.name = 'unnamed_asset'
            
        # Convert to percentage for numerical stability
        returns_pct = series * 100
    
    # ... rest of the method remains the same
        
        # Set up mean model
        if mean == 'zero':
            mean_model = ZeroMean(returns_pct)
        else:
            mean_model = ConstantMean(returns_pct)
        
        # Set up volatility model
        if model_type == 'EGARCH':
            vol_model = EGARCH(p=p, q=q)
        elif model_type == 'GJR-GARCH':
            vol_model = GARCH(p=p, o=o, q=q)
        else:
            vol_model = GARCH(p=p, q=q)
        
        # Combine mean and volatility models
        model = mean_model
        model.volatility = vol_model
        # Set distribution using the correct method
        from arch.univariate import Normal, StudentsT, SkewStudent
        if dist == 'normal':
            model.distribution = Normal()
        elif dist == 't':
            model.distribution = StudentsT()
        elif dist == 'skewt':
            model.distribution = SkewStudent()

        
        # Fit the model
        logger.info(f"Fitting {model_type} model...")
        try:
            res = model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            params = {
                'omega': res.params.get('omega', np.nan),
                'alpha': res.params.get('alpha[1]', np.nan),
                'beta': res.params.get('beta[1]', np.nan)
            }
            
            if model_type == 'GJR-GARCH':
                params['gamma'] = res.params.get('gamma[1]', np.nan)
            
            if dist == 't':
                params['nu'] = res.params.get('nu', np.nan)
            
            # Get conditional volatility (convert back to decimal)
            conditional_vol = res.conditional_volatility / 100
            
            # Store results
            result = GARCHResults(
                model_type=model_type,
                parameters=params,
                conditional_volatility=conditional_vol,
                volatility_forecast=None,  # Will be set by forecast method
                forecast_horizon=0,
                aic=res.aic,
                bic=res.bic,
                log_likelihood=res.loglikelihood
            )
            
            # Store fitted model for forecasting
            self.fitted_models[series.name] = res
            self.results[series.name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error fitting {model_type}: {str(e)}")
            raise
    
    def forecast_volatility(
        self,
        asset: str,
        horizon: int = 20,
        n_simulations: int = 1000
    ) -> np.ndarray:
        """
        Generate volatility forecasts.
        
        Args:
            asset: Asset name
            horizon: Forecast horizon in days
            n_simulations: Number of simulations for confidence intervals
            
        Returns:
            Array of volatility forecasts
        """
        if asset not in self.fitted_models:
            raise ValueError(f"No fitted model for {asset}")
        
        model_result = self.fitted_models[asset]
        
        # Generate forecasts
        forecast = model_result.forecast(horizon=horizon)
        variance_forecast = forecast.variance.values[-1, :]
        
        # Convert to volatility and back to decimal form
        volatility_forecast = np.sqrt(variance_forecast) / 100
        
        # Update results
        self.results[asset].volatility_forecast = volatility_forecast
        self.results[asset].forecast_horizon = horizon
        
        return volatility_forecast
    
    def fit_all_assets(
        self,
        model_type: str = 'GARCH',
        **kwargs
    ) -> Dict[str, GARCHResults]:
        """
        Fit GARCH models to all assets in the dataset.
        
        Args:
            model_type: Type of GARCH model
            **kwargs: Additional arguments for fit_garch
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        if isinstance(self.returns_data, pd.Series):
            # Single asset
            result = self.fit_garch(self.returns_data, model_type, **kwargs)
            results[self.returns_data.name] = result
        else:
            # Multiple assets
            for asset in self.returns_data.columns:
                logger.info(f"Fitting {model_type} for {asset}")
                result = self.fit_garch(
                    self.returns_data[asset], 
                    model_type, 
                    **kwargs
                )
                results[asset] = result
        
        return results
    
    def calculate_risk_parity_weights(
        self,
        forecast_horizon: int = 20
    ) -> pd.Series:
        """
        Calculate risk parity weights based on volatility forecasts.
        
        Risk parity allocates capital inversely proportional to volatility
        so each asset contributes equally to portfolio risk.
        
        Args:
            forecast_horizon: Days ahead to forecast
            
        Returns:
            Series of portfolio weights
        """
        volatilities = {}
        
        for asset in self.returns_data.columns:
            if asset not in self.fitted_models:
                self.fit_garch(self.returns_data[asset])
            
            # Get volatility forecast
            vol_forecast = self.forecast_volatility(asset, horizon=forecast_horizon)
            
            # Use average volatility over forecast horizon
            volatilities[asset] = vol_forecast.mean()
        
        # Calculate inverse volatility weights
        vols = pd.Series(volatilities)
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        
        return weights
    
    def calculate_min_variance_weights(
        self,
        covariance_matrix: Optional[pd.DataFrame] = None,
        use_garch_vol: bool = True
    ) -> pd.Series:
        """
        Calculate minimum variance weights using GARCH volatilities.
        
        Args:
            covariance_matrix: Optional covariance matrix
            use_garch_vol: Whether to use GARCH volatilities
            
        Returns:
            Series of portfolio weights
        """
        if covariance_matrix is None:
            # Build covariance matrix using GARCH volatilities
            n_assets = len(self.returns_data.columns)
            
            if use_garch_vol:
                # Get GARCH volatilities
                vols = []
                for asset in self.returns_data.columns:
                    if asset in self.results:
                        vol = self.results[asset].conditional_volatility.iloc[-1]
                    else:
                        vol = self.returns_data[asset].std()
                    vols.append(vol)
                
                # Use correlation matrix with GARCH volatilities
                corr_matrix = self.returns_data.corr()
                vol_diag = np.diag(vols)
                cov_matrix = vol_diag @ corr_matrix @ vol_diag
            else:
                cov_matrix = self.returns_data.cov()
            
            covariance_matrix = pd.DataFrame(
                cov_matrix,
                index=self.returns_data.columns,
                columns=self.returns_data.columns
            )
        
        try:
            # Add small regularization to ensure positive definite
            reg_matrix = covariance_matrix + np.eye(len(covariance_matrix)) * 1e-8
            
            # Solve for minimum variance weights
            inv_cov = np.linalg.inv(reg_matrix)
            ones = np.ones(len(covariance_matrix))
            
            weights = inv_cov @ ones / (ones @ inv_cov @ ones)
            
            # Ensure weights sum to exactly 1.0 (handle floating point precision)
            weights = weights / weights.sum()
            
            # Check for NaN values
            if np.any(np.isnan(weights)):
                # Fallback to equal weights
                logger.warning("NaN weights detected, using equal weights")
                weights = np.ones(len(covariance_matrix)) / len(covariance_matrix)
                
        except np.linalg.LinAlgError:
            # If matrix is singular, use equal weights
            logger.warning("Singular covariance matrix, using equal weights")
            weights = np.ones(len(covariance_matrix)) / len(covariance_matrix)
        
        return pd.Series(weights, index=covariance_matrix.index)
    
    def detect_volatility_regime(
        self,
        asset: str,
        low_threshold: float = 0.25,
        high_threshold: float = 0.75
    ) -> str:
        """
        Detect current volatility regime.
        
        Args:
            asset: Asset name
            low_threshold: Percentile for low volatility
            high_threshold: Percentile for high volatility
            
        Returns:
            'low', 'normal', or 'high'
        """
        if asset not in self.results:
            raise ValueError(f"No results for {asset}")
        
        conditional_vol = self.results[asset].conditional_volatility
        
        # Calculate percentiles
        current_vol = conditional_vol.iloc[-1]
        low_cutoff = conditional_vol.quantile(low_threshold)
        high_cutoff = conditional_vol.quantile(high_threshold)
        
        if current_vol <= low_cutoff:
            return 'low'
        elif current_vol >= high_cutoff:
            return 'high'
        else:
            return 'normal'
    
    def calculate_var_cvar(
        self,
        weights: pd.Series,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate VaR and CVaR using GARCH volatility forecasts.
        
        Args:
            weights: Portfolio weights
            confidence_level: Confidence level (e.g., 0.95)
            horizon: Time horizon in days
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        # Get portfolio returns
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        portfolio_returns.name = 'portfolio'  # Give it a name
        
        # Fit GARCH to portfolio returns
        portfolio_garch = self.fit_garch(
            portfolio_returns,
            model_type='GARCH',
            dist='t'
        )
        
        # Get volatility forecast
        # The portfolio is now in fitted_models with key 'portfolio'
        vol_forecast = self.forecast_volatility('portfolio', horizon=horizon)
        
        # Calculate VaR and CVaR
        from scipy import stats
        
        if 'nu' in portfolio_garch.parameters:
            # Student-t distribution
            nu = portfolio_garch.parameters['nu']
            z_score = stats.t.ppf(1 - confidence_level, nu)
        else:
            # Normal distribution
            z_score = stats.norm.ppf(1 - confidence_level)
        
        # Multi-period VaR
        if horizon > 1:
            total_vol = np.sqrt(vol_forecast[:horizon].sum())
        else:
            total_vol = vol_forecast[0]
        
        var = portfolio_returns.mean() + z_score * total_vol
        
        # CVaR approximation
        if 'nu' in portfolio_garch.parameters:
            cvar_multiplier = stats.t.pdf(z_score, nu) / (1 - confidence_level)
            cvar = portfolio_returns.mean() - total_vol * cvar_multiplier * np.sqrt((nu + z_score**2) / (nu - 1))
        else:
            cvar_multiplier = stats.norm.pdf(z_score) / (1 - confidence_level)
            cvar = portfolio_returns.mean() - total_vol * cvar_multiplier
        
        return var, cvar
        
    def compare_models(self) -> pd.DataFrame:
        """
        Compare different GARCH specifications for all assets.
        
        Returns:
            DataFrame with model comparison metrics
        """
        model_types = ['GARCH', 'GJR-GARCH', 'EGARCH']
        comparison_results = []
        
        for asset in self.returns_data.columns:
            for model_type in model_types:
                try:
                    result = self.fit_garch(
                        self.returns_data[asset],
                        model_type=model_type,
                        dist='t'
                    )
                    
                    comparison_results.append({
                        'Asset': asset,
                        'Model': model_type,
                        'AIC': result.aic,
                        'BIC': result.bic,
                        'Log-Likelihood': result.log_likelihood,
                        'Volatility': result.conditional_volatility.iloc[-1]
                    })
                except:
                    continue
        
        return pd.DataFrame(comparison_results)


class DynamicGARCHStrategy:
    """
    Dynamic portfolio strategy using GARCH volatility forecasts.
    
    This class implements a trading strategy that adjusts portfolio
    weights based on volatility regimes and forecasts.
    """
    
    def __init__(
        self,
        garch_modeler: GARCHModeler,
        base_weights: Optional[pd.Series] = None
    ):
        """
        Initialize dynamic strategy.
        
        Args:
            garch_modeler: Fitted GARCHModeler instance
            base_weights: Base portfolio weights (default: equal weight)
        """
        self.garch_modeler = garch_modeler
        self.base_weights = base_weights
        
        if self.base_weights is None:
            n_assets = len(garch_modeler.returns_data.columns)
            self.base_weights = pd.Series(
                1/n_assets,
                index=garch_modeler.returns_data.columns
            )
    
    def calculate_dynamic_weights(
        self,
        method: str = 'risk_parity',
        volatility_target: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Calculate dynamic portfolio weights.
        
        Args:
            method: 'risk_parity', 'min_variance', 'volatility_scaling'
            volatility_target: Target portfolio volatility
            **kwargs: Additional method-specific arguments
            
        Returns:
            Series of portfolio weights
        """
        if method == 'risk_parity':
            return self.garch_modeler.calculate_risk_parity_weights(**kwargs)
        
        elif method == 'min_variance':
            return self.garch_modeler.calculate_min_variance_weights(**kwargs)
        
        elif method == 'volatility_scaling':
            # Scale weights based on volatility regime
            weights = self.base_weights.copy()
            
            for asset in weights.index:
                regime = self.garch_modeler.detect_volatility_regime(asset)
                
                if regime == 'high':
                    weights[asset] *= 0.5  # Reduce exposure
                elif regime == 'low':
                    weights[asset] *= 1.5  # Increase exposure
            
            # Renormalize
            weights = weights / weights.sum()
            
            # Apply volatility target if specified
            if volatility_target:
                current_vol = self._calculate_portfolio_volatility(weights)
                scale_factor = volatility_target / current_vol
                weights = weights * min(scale_factor, 2.0)  # Cap leverage
            
            return weights
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _calculate_portfolio_volatility(self, weights: pd.Series) -> float:
        """Calculate portfolio volatility using GARCH estimates."""
        portfolio_returns = (self.garch_modeler.returns_data * weights).sum(axis=1)
        return portfolio_returns.std() * np.sqrt(252)
    
    def backtest_strategy(
        self,
        rebalance_frequency: str = 'monthly',
        method: str = 'risk_parity',
        **kwargs
    ) -> Dict[str, Union[pd.Series, float]]:
        """
        Backtest the dynamic GARCH strategy.
        
        Args:
            rebalance_frequency: 'daily', 'weekly', 'monthly'
            method: Weight calculation method
            **kwargs: Additional arguments for weight calculation
            
        Returns:
            Dictionary with backtest results
        """
        returns_data = self.garch_modeler.returns_data
        dates = returns_data.index
        
        # Set rebalance dates
        if rebalance_frequency == 'daily':
            rebalance_dates = dates[252:]  # Start after 1 year
        elif rebalance_frequency == 'weekly':
            rebalance_dates = dates[252::5]
        elif rebalance_frequency == 'monthly':
            rebalance_dates = dates[252::21]
        else:
            raise ValueError(f"Unknown frequency: {rebalance_frequency}")
        
        # Initialize results
        portfolio_returns = []
        weights_history = []
        
        # Initial weights
        current_weights = self.base_weights
        
        for i, date in enumerate(dates[252:]):
            # Calculate portfolio return
            daily_return = (returns_data.loc[date] * current_weights).sum()
            portfolio_returns.append(daily_return)
            
            # Rebalance if needed
            if date in rebalance_dates:
                # Refit GARCH models with data up to current date
                historical_data = returns_data.loc[:date]
                self.garch_modeler.returns_data = historical_data
                self.garch_modeler.fit_all_assets()
                
                # Calculate new weights
                current_weights = self.calculate_dynamic_weights(method, **kwargs)
                weights_history.append({
                    'date': date,
                    'weights': current_weights.to_dict()
                })
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_returns, index=dates[252:])
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol
        
        # Maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'weights_history': pd.DataFrame(weights_history),
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }