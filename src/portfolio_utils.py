"""
Portfolio utilities module.

This module contains common portfolio calculation functions used across
the project, including optimization helpers, risk metrics, and performance
calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    asset_names: List[str]
    
    def get_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation as a dictionary."""
        return dict(zip(self.asset_names, self.weights))
    
    def to_series(self) -> pd.Series:
        """Convert weights to pandas Series."""
        return pd.Series(self.weights, index=self.asset_names)


def calculate_portfolio_stats(
    weights: np.ndarray,
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame]
) -> Tuple[float, float]:
    """
    Calculate portfolio expected return and volatility.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix of returns
        
    Returns:
        Tuple of (expected_return, volatility)
    """
    # Convert pandas objects to numpy if needed
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Portfolio return is weighted average of individual returns
    portfolio_return = np.sum(weights * expected_returns)
    
    # Portfolio variance: w^T * Σ * w
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    return portfolio_return, portfolio_volatility


def calculate_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio for given weights."""
    p_return, p_volatility = calculate_portfolio_stats(weights, expected_returns, cov_matrix)
    return (p_return - risk_free_rate) / p_volatility


def maximize_sharpe_ratio(
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.02,
    constraints: Optional[Dict] = None,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> Tuple[np.ndarray, float]:
    """
    Maximize Sharpe ratio using multiple starting points to avoid local optima.
    
    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix of returns
        risk_free_rate: Risk-free rate for Sharpe calculation
        constraints: Additional constraints for optimization
        bounds: Bounds for each weight (default: (0, 1))
        
    Returns:
        Tuple of (optimal_weights, sharpe_ratio)
    """
    n_assets = len(expected_returns)
    
    # Convert to numpy if needed
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Objective function (negative Sharpe for minimization)
    def negative_sharpe(weights):
        p_return = np.dot(weights, expected_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_return - risk_free_rate) / p_vol
    
    # Default constraints
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    elif not isinstance(constraints, list):
        constraints = [constraints]
    
    # Always add sum-to-one constraint
    sum_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    if sum_constraint not in constraints:
        constraints.append(sum_constraint)
    
    # Default bounds
    if bounds is None:
        bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Try multiple starting points
    best_result = None
    best_sharpe = -np.inf
    
    # Test 10 random starting points plus equal weights
    for i in range(11):
        if i < 10:
            # Random starting weights
            x0 = np.random.random(n_assets)
            x0 = x0 / np.sum(x0)
        else:
            # Equal weights as final attempt
            x0 = np.ones(n_assets) / n_assets
        
        # Run optimization
        try:
            result = minimize(
                negative_sharpe,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )
            
            if result.success:
                sharpe = -result.fun
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result
        except:
            continue
    
    if best_result is None:
        raise ValueError("Optimization failed to converge")
    
    return best_result.x, best_sharpe


def minimize_volatility(
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    constraints: Optional[Dict] = None,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> Tuple[np.ndarray, float]:
    """
    Find minimum volatility portfolio.
    
    Args:
        expected_returns: Expected returns (used for constraints)
        cov_matrix: Covariance matrix of returns
        constraints: Additional constraints
        bounds: Bounds for each weight
        
    Returns:
        Tuple of (optimal_weights, volatility)
    """
    n_assets = len(expected_returns)
    
    # Convert to numpy if needed
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Objective function
    def portfolio_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Constraints
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Bounds
    if bounds is None:
        bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess
    x0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        portfolio_vol,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    if not result.success:
        raise ValueError("Minimum volatility optimization failed")
    
    return result.x, portfolio_vol(result.x)


def calculate_risk_parity_weights(
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    bounds: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Calculate risk parity weights where each asset contributes equally to risk.
    
    Args:
        cov_matrix: Covariance matrix of returns
        bounds: Bounds for each weight
        
    Returns:
        Risk parity weights
    """
    n_assets = cov_matrix.shape[0]
    
    # Convert to numpy if needed
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Objective: minimize difference between risk contributions
    def objective(weights, cov_matrix):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights)
        contrib = weights * marginal_contrib / portfolio_vol
        
        # Target equal contribution (1/n for each asset)
        target_contrib = np.ones(n_assets) / n_assets
        return np.sum((contrib - target_contrib) ** 2)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Bounds
    if bounds is None:
        bounds = tuple((0.01, 1) for _ in range(n_assets))
    
    # Initial guess - inverse volatility weighted
    vols = np.sqrt(np.diag(cov_matrix))
    x0 = (1/vols) / np.sum(1/vols)
    
    # Optimize
    result = minimize(
        objective,
        x0=x0,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    
    return result.x


def calculate_risk_contributions(
    weights: np.ndarray,
    cov_matrix: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Calculate risk contribution of each asset to portfolio risk.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        
    Returns:
        Array of risk contributions (sums to 1)
    """
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    marginal_contrib = np.dot(cov_matrix, weights)
    contrib = weights * marginal_contrib / portfolio_var
    
    return contrib / contrib.sum()  # Normalize to sum to 1


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Series of drawdowns (negative values)
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    return calculate_drawdown(returns).min()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    expected_return = returns.mean() * periods_per_year
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    return (expected_return - risk_free_rate) / downside_std


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * periods_per_year
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return np.inf
    
    return annual_return / max_dd


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
        
    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    
    if active_returns.std() == 0:
        return 0
    
    return (active_returns.mean() * periods_per_year) / (active_returns.std() * np.sqrt(periods_per_year))


def calculate_turnover(
    weights_history: pd.DataFrame,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annual portfolio turnover.
    
    Args:
        weights_history: DataFrame with weight history (assets as columns)
        periods_per_year: Number of periods per year
        
    Returns:
        Annual turnover rate
    """
    weight_changes = weights_history.diff().abs().sum(axis=1)
    avg_daily_turnover = weight_changes.mean() / 2  # Divide by 2 to avoid double counting
    annual_turnover = avg_daily_turnover * periods_per_year
    
    return annual_turnover


def calculate_efficient_frontier(
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    n_points: int = 100,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Calculate efficient frontier points.
    
    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        n_points: Number of points on the frontier
        bounds: Weight bounds
        
    Returns:
        Tuple of (volatilities, returns, weights_list)
    """
    n_assets = len(expected_returns)
    
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Target returns
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    # Storage
    frontier_volatility = []
    frontier_returns = []
    frontier_weights = []
    
    # Bounds
    if bounds is None:
        bounds = tuple((0, 1) for _ in range(n_assets))
    
    for target in target_returns:
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, t=target: np.dot(x, expected_returns) - t}
        ]
        
        # Minimize volatility for target return
        result = minimize(
            lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
            x0=np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )
        
        if result.success:
            ret, vol = calculate_portfolio_stats(result.x, expected_returns, cov_matrix)
            frontier_returns.append(ret)
            frontier_volatility.append(vol)
            frontier_weights.append(result.x)
    
    return np.array(frontier_volatility), np.array(frontier_returns), frontier_weights


# Portfolio performance metrics calculation
def calculate_portfolio_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio performance metrics.
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary of performance metrics
    """
    # Annualized metrics
    annual_return = returns.mean() * periods_per_year
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # Risk metrics
    max_dd = calculate_max_drawdown(returns)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(returns, periods_per_year)
    
    # Other metrics
    skew = returns.skew()
    kurt = returns.kurtosis()
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'skewness': skew,
        'kurtosis': kurt,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': (returns > 0).mean(),
        'best_day': returns.max(),
        'worst_day': returns.min()
    }