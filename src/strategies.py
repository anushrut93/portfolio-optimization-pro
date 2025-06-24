"""
Portfolio strategy implementations.

This module contains various portfolio optimization strategies including
Black-Litterman, ML-enhanced strategies, and strategy factory functions
for backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy.optimize import minimize
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

from .portfolio_utils import (
    calculate_portfolio_stats,
    maximize_sharpe_ratio,
    minimize_volatility,
    calculate_risk_parity_weights,
    OptimizationResult
)


class BlackLittermanModel:
    """
    Implementation of the Black-Litterman model for combining market equilibrium
    with investor views to generate enhanced return estimates.
    
    The Black-Litterman model addresses a key weakness of mean-variance optimization:
    the sensitivity to expected return estimates. It starts with equilibrium returns
    implied by market weights and allows investors to blend in their views.
    """
    
    def __init__(self, 
                 cov_matrix: pd.DataFrame,
                 market_weights: Optional[pd.Series] = None,
                 risk_aversion: float = 2.5,
                 tau: float = 0.05):
        """
        Initialize Black-Litterman model.
        
        Args:
            cov_matrix: Covariance matrix of returns
            market_weights: Market capitalization weights (if None, uses equal weights)
            risk_aversion: Market risk aversion parameter (typically 2-3)
            tau: Uncertainty scaling parameter (typically 0.01-0.1)
        """
        self.cov_matrix = cov_matrix
        self.tau = tau
        self.risk_aversion = risk_aversion
        
        # Use equal weights if market weights not provided
        if market_weights is None:
            n_assets = len(cov_matrix)
            self.market_weights = pd.Series(
                np.ones(n_assets) / n_assets,
                index=cov_matrix.index
            )
        else:
            self.market_weights = market_weights
            
        # Calculate equilibrium returns
        self.equilibrium_returns = self._calculate_equilibrium_returns()
    
    def _calculate_equilibrium_returns(self) -> pd.Series:
        """
        Calculate market equilibrium returns using reverse optimization.
        
        Pi = delta * Sigma * w_mkt
        where delta is risk aversion and w_mkt are market weights
        """
        pi = self.risk_aversion * self.cov_matrix.dot(self.market_weights)
        return pd.Series(pi, index=self.cov_matrix.index)
    
    def add_views(self, 
                  views: Dict[str, float],
                  confidence: Union[float, Dict[str, float]] = 1.0) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Add investor views and calculate posterior returns.
        
        Args:
            views: Dictionary of {asset: expected_return} for assets with views
            confidence: Confidence in views (1.0 = high confidence, larger = less confidence)
                       Can be a single value or dict per view
                       
        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        
        Example:
            views = {'AAPL': 0.15, 'MSFT': 0.12}  # Expect 15% return for AAPL, 12% for MSFT
            posterior_returns, posterior_cov = model.add_views(views, confidence=0.8)
        """
        assets = list(self.cov_matrix.index)
        n_assets = len(assets)
        k_views = len(views)
        
        # Build P matrix (which assets views are about)
        P = np.zeros((k_views, n_assets))
        Q = np.zeros(k_views)
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.index(asset)
                P[i, asset_idx] = 1
                Q[i] = view_return
        
        # Build Omega (uncertainty about views)
        if isinstance(confidence, dict):
            # Different confidence per view
            omega_diag = []
            for asset in views.keys():
                conf = confidence.get(asset, 1.0)
                omega_diag.append(self.tau * conf)
            Omega = np.diag(omega_diag)
        else:
            # Same confidence for all views
            Omega = np.eye(k_views) * self.tau * confidence
        
        # Calculate posterior returns using Black-Litterman formula
        # First calculate the posterior covariance
        tau_sigma = self.tau * self.cov_matrix.values
        
        # Posterior covariance
        inv_tau_sigma = linalg.inv(tau_sigma)
        inv_omega = linalg.inv(Omega)
        posterior_cov_inv = inv_tau_sigma + P.T.dot(inv_omega).dot(P)
        posterior_cov = linalg.inv(posterior_cov_inv)
        
        # Posterior returns
        posterior_returns = posterior_cov.dot(
            inv_tau_sigma.dot(self.equilibrium_returns.values) + 
            P.T.dot(inv_omega).dot(Q)
        )
        
        # Convert to pandas objects
        posterior_returns = pd.Series(posterior_returns, index=assets)
        posterior_cov = pd.DataFrame(posterior_cov, index=assets, columns=assets)
        
        return posterior_returns, posterior_cov
    
    def get_optimal_weights(self,
                           views: Dict[str, float],
                           confidence: Union[float, Dict[str, float]] = 1.0,
                           risk_free_rate: float = 0.02) -> pd.Series:
        """
        Get optimal portfolio weights using Black-Litterman posterior returns.
        
        Args:
            views: Dictionary of views
            confidence: Confidence in views
            risk_free_rate: Risk-free rate for Sharpe optimization
            
        Returns:
            Optimal portfolio weights
        """
        # Get posterior returns
        posterior_returns, posterior_cov = self.add_views(views, confidence)
        
        # Optimize using posterior returns
        weights, _ = maximize_sharpe_ratio(
            posterior_returns,
            posterior_cov,
            risk_free_rate
        )
        
        return pd.Series(weights, index=posterior_returns.index)


def create_market_cap_weights(prices: pd.DataFrame, 
                             shares_outstanding: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Create market capitalization weights from prices.
    
    Args:
        prices: DataFrame of asset prices
        shares_outstanding: Dict of shares outstanding per asset
                          If None, assumes equal shares (simplified)
                          
    Returns:
        Market cap weights as Series
    """
    latest_prices = prices.iloc[-1]
    
    if shares_outstanding is None:
        # Simplified: assume relative prices indicate relative market caps
        # In reality, you'd need actual shares outstanding data
        market_caps = latest_prices / latest_prices.sum()
    else:
        market_caps = pd.Series(
            {asset: latest_prices[asset] * shares_outstanding.get(asset, 1)
             for asset in latest_prices.index}
        )
        market_caps = market_caps / market_caps.sum()
    
    return market_caps


def calculate_dynamic_black_litterman_weights(
    historical_prices: pd.DataFrame,
    lookback_days: int = 60,
    momentum_threshold: float = 0.05,
    confidence_base: float = 1.0) -> np.ndarray:
    """
    Calculate Black-Litterman weights with dynamic views based on momentum.
    
    This function creates views based on recent price momentum and applies
    the Black-Litterman model to get optimal weights.
    
    Args:
        historical_prices: Recent price history
        lookback_days: Days to calculate momentum
        momentum_threshold: Minimum momentum to create a view
        confidence_base: Base confidence in views
        
    Returns:
        Array of optimal weights
    """
    # Use most recent data
    recent_prices = historical_prices.tail(lookback_days)
    
    # Calculate returns and covariance
    returns = recent_prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Calculate momentum (simple return over period)
    momentum = (recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1
    
    # Create views based on momentum
    views = {}
    confidence = {}
    
    for asset in momentum.index:
        if abs(momentum[asset]) > momentum_threshold:
            # Strong momentum creates a view
            # Project momentum forward (simplified)
            annual_view = momentum[asset] * (252 / lookback_days)
            views[asset] = annual_view
            
            # Higher confidence for stronger momentum
            confidence[asset] = confidence_base / (1 + abs(momentum[asset]))
    
    # If no strong views, use equal weights
    if not views:
        return np.ones(len(historical_prices.columns)) / len(historical_prices.columns)
    
    # Create Black-Litterman model
    model = BlackLittermanModel(cov_matrix)
    
    # Get optimal weights
    try:
        weights = model.get_optimal_weights(views, confidence)
        return weights.values
    except:
        # Fallback to equal weights if optimization fails
        return np.ones(len(historical_prices.columns)) / len(historical_prices.columns)


# Strategy factory functions for backtesting
def create_max_sharpe_strategy(risk_free_rate: float = 0.02) -> Callable:
    """
    Create a maximum Sharpe ratio strategy function.
    
    Args:
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Strategy function that takes historical prices and returns weights
    """
    def max_sharpe_strategy(historical_prices: pd.DataFrame) -> np.ndarray:
        returns = historical_prices.pct_change().dropna()
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        weights, _ = maximize_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate)
        return weights
    
    return max_sharpe_strategy


def create_min_volatility_strategy() -> Callable:
    """Create a minimum volatility strategy function."""
    def min_vol_strategy(historical_prices: pd.DataFrame) -> np.ndarray:
        returns = historical_prices.pct_change().dropna()
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        weights, _ = minimize_volatility(expected_returns, cov_matrix)
        return weights
    
    return min_vol_strategy


def create_risk_parity_strategy() -> Callable:
    """Create a risk parity strategy function."""
    def risk_parity_strategy(historical_prices: pd.DataFrame) -> np.ndarray:
        returns = historical_prices.pct_change().dropna()
        cov_matrix = returns.cov() * 252
        
        weights = calculate_risk_parity_weights(cov_matrix)
        return weights
    
    return risk_parity_strategy


def create_equal_weight_strategy() -> Callable:
    """Create an equal weight strategy function."""
    def equal_weight_strategy(historical_prices: pd.DataFrame) -> np.ndarray:
        n_assets = len(historical_prices.columns)
        return np.ones(n_assets) / n_assets
    
    return equal_weight_strategy


def create_black_litterman_strategy(
    views: Dict[str, float],
    confidence: Union[float, Dict[str, float]] = 1.0,
    risk_free_rate: float = 0.02) -> Callable:
    """
    Create a static Black-Litterman strategy with fixed views.
    
    Args:
        views: Fixed views to use
        confidence: Confidence in views
        risk_free_rate: Risk-free rate
        
    Returns:
        Strategy function
    """
    def black_litterman_strategy(historical_prices: pd.DataFrame) -> np.ndarray:
        returns = historical_prices.pct_change().dropna()
        cov_matrix = returns.cov() * 252
        
        model = BlackLittermanModel(cov_matrix)
        weights = model.get_optimal_weights(views, confidence, risk_free_rate)
        
        return weights.values
    
    return black_litterman_strategy


def create_dynamic_black_litterman_strategy(
    lookback_days: int = 60,
    momentum_threshold: float = 0.05) -> Callable:
    """
    Create a dynamic Black-Litterman strategy that updates views based on momentum.
    
    Args:
        lookback_days: Days to calculate momentum
        momentum_threshold: Minimum momentum for views
        
    Returns:
        Strategy function
    """
    def dynamic_bl_strategy(historical_prices: pd.DataFrame) -> np.ndarray:
        return calculate_dynamic_black_litterman_weights(
            historical_prices,
            lookback_days,
            momentum_threshold
        )
    
    return dynamic_bl_strategy


def create_ml_enhanced_strategy(
    ml_predictions: pd.Series,
    blend_factor: float = 0.6,
    risk_free_rate: float = 0.02,
    constraints: Optional[Dict] = None) -> Callable:
    """
    Create an ML-enhanced strategy that blends ML predictions with historical returns.
    
    Args:
        ml_predictions: ML model's return predictions
        blend_factor: Weight on ML predictions (0-1)
        risk_free_rate: Risk-free rate
        constraints: Optional optimization constraints
        
    Returns:
        Strategy function
    """
    def ml_enhanced_strategy(historical_prices: pd.DataFrame) -> np.ndarray:
        returns = historical_prices.pct_change().dropna()
        historical_mean = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Blend ML predictions with historical
        blended_returns = blend_factor * ml_predictions + (1 - blend_factor) * historical_mean
        
        # Ensure same assets
        common_assets = list(set(historical_mean.index) & set(ml_predictions.index))
        blended_returns = blended_returns[common_assets]
        cov_matrix = cov_matrix.loc[common_assets, common_assets]
        
        weights, _ = maximize_sharpe_ratio(
            blended_returns, 
            cov_matrix, 
            risk_free_rate,
            constraints
        )
        
        # Map back to original order
        weight_series = pd.Series(weights, index=common_assets)
        final_weights = np.zeros(len(historical_prices.columns))
        for i, asset in enumerate(historical_prices.columns):
            if asset in weight_series:
                final_weights[i] = weight_series[asset]
        
        # Renormalize if needed
        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum()
        else:
            final_weights = np.ones(len(final_weights)) / len(final_weights)
            
        return final_weights
    
    return ml_enhanced_strategy