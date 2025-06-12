"""
Mean-Variance Portfolio Optimization Module.

This module implements Modern Portfolio Theory (MPT) using the Markowitz
mean-variance optimization framework. It finds the optimal weights for
a portfolio of assets that maximizes return for a given level of risk
or minimizes risk for a given level of return.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import scipy.optimize as sco
from dataclasses import dataclass
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Container for portfolio optimization results.
    
    This class holds all the important information about an optimized portfolio,
    making it easy to access and understand the results.
    """
    weights: np.ndarray          # The optimal weight for each asset
    expected_return: float       # Expected annual return
    volatility: float           # Expected annual volatility (risk)
    sharpe_ratio: float         # Risk-adjusted return metric
    asset_names: List[str]      # Names of assets in the portfolio
    
    def get_allocation(self) -> Dict[str, float]:
        """Get the allocation as a dictionary mapping asset names to weights."""
        return dict(zip(self.asset_names, self.weights))
    
    def __str__(self) -> str:
        """Create a readable string representation of the results."""
        allocation_str = "\n".join([
            f"  {asset}: {weight:.1%}" 
            for asset, weight in self.get_allocation().items()
            if weight > 0.001  # Only show positions > 0.1%
        ])
        
        return f"""
Portfolio Optimization Results:
------------------------------
Expected Annual Return: {self.expected_return:.1%}
Annual Volatility: {self.volatility:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}

Asset Allocation:
{allocation_str}
"""


class MeanVarianceOptimizer:
    """
    Implements mean-variance portfolio optimization.
    
    This optimizer uses historical returns to estimate expected returns and
    covariance, then finds optimal portfolio weights using various optimization
    objectives (maximum Sharpe ratio, minimum volatility, etc.).
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
                          (default: 2% annually)
        """
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.asset_names = None
        
    def calculate_portfolio_stats(
        self, 
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate portfolio expected return and volatility.
        
        This is the core calculation that tells us how a portfolio with given
        weights would perform based on historical data.
        
        Args:
            weights: Array of portfolio weights (must sum to 1)
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Tuple of (expected_return, volatility)
        """
        # Portfolio return is weighted average of individual returns
        portfolio_return = np.sum(weights * expected_returns)
        
        # Portfolio variance requires matrix multiplication
        # variance = w^T * Σ * w (where Σ is covariance matrix)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio for optimization.
        
        We use negative because scipy minimizes functions, but we want to
        maximize the Sharpe ratio. This is a common trick in optimization.
        """
        p_return, p_volatility = self.calculate_portfolio_stats(
            weights, self.expected_returns, self.cov_matrix
        )
        
        # Sharpe ratio = (return - risk_free_rate) / volatility
        # We return negative to convert maximization to minimization
        return -(p_return - self.risk_free_rate) / p_volatility
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility for minimum variance optimization."""
        _, p_volatility = self.calculate_portfolio_stats(
            weights, self.expected_returns, self.cov_matrix
        )
        return p_volatility
    
    def prepare_data(self, prices: pd.DataFrame) -> None:
        """
        Prepare price data for optimization.
        
        This method calculates returns, expected returns, and the covariance
        matrix from historical price data.
        
        Args:
            prices: DataFrame with asset prices (assets as columns)
        """
        # Calculate daily returns using logarithmic returns
        # Log returns are better for statistical properties
        self.returns = np.log(prices / prices.shift(1)).dropna()
        
        # Store asset names for later reference
        self.asset_names = list(prices.columns)
        
        # Calculate expected returns (annualized)
        # We use 252 trading days per year
        self.expected_returns = self.returns.mean() * 252
        
        # Calculate covariance matrix (annualized)
        self.cov_matrix = self.returns.cov() * 252
        
        logger.info(f"Prepared data for {len(self.asset_names)} assets")
        logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        logger.info(f"Expected returns range: {self.expected_returns.min():.1%} to {self.expected_returns.max():.1%}")
    
    def optimize(
        self,
        prices: pd.DataFrame,
        objective: str = 'max_sharpe',
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights based on the specified objective.
        
        Args:
            prices: DataFrame with historical prices
            objective: Optimization objective ('max_sharpe', 'min_volatility')
            constraints: Optional constraints on weights
            
        Returns:
            OptimizationResult object with optimal weights and statistics
        """
        # Prepare the data
        self.prepare_data(prices)
        
        # Number of assets
        n_assets = len(self.asset_names)
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Constraints: weights must sum to 1 (fully invested)
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds: each weight between 0 and 1 (no short selling by default)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Add any user-specified constraints
        if constraints:
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                bounds = tuple((0, max_weight) for _ in range(n_assets))
                
        # Choose objective function
        if objective == 'max_sharpe':
            objective_function = self.negative_sharpe_ratio
        elif objective == 'min_volatility':
            objective_function = self.portfolio_volatility
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Run the optimization
        logger.info(f"Running optimization with objective: {objective}")
        result = sco.minimize(
            objective_function,
            initial_weights,
            method='SLSQP',  # Sequential Least Squares Programming
            bounds=bounds,
            constraints=constraints_list,
            options={'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate statistics for optimal portfolio
        expected_return, volatility = self.calculate_portfolio_stats(
            optimal_weights, self.expected_returns, self.cov_matrix
        )
        
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        return OptimizationResult(
            weights=optimal_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            asset_names=self.asset_names
        )
    
    def efficient_frontier(
        self,
        prices: pd.DataFrame,
        n_portfolios: int = 100
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.
        
        The efficient frontier represents the set of optimal portfolios that
        offer the highest expected return for each level of risk.
        
        Args:
            prices: DataFrame with historical prices
            n_portfolios: Number of points on the frontier
            
        Returns:
            DataFrame with frontier portfolios
        """
        self.prepare_data(prices)
        
        # Find the minimum and maximum possible returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        
        # Create target returns evenly spaced between min and max
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        frontier_volatilities = []
        frontier_returns = []
        frontier_weights = []
        
        for target_return in target_returns:
            # Add constraint for target return
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, r=target_return: 
                 np.sum(x * self.expected_returns) - r}
            ]
            
            # Optimize for minimum volatility given target return
            result = sco.minimize(
                self.portfolio_volatility,
                np.array([1/len(self.asset_names)] * len(self.asset_names)),
                method='SLSQP',
                bounds=tuple((0, 1) for _ in range(len(self.asset_names))),
                constraints=constraints,
                options={'disp': False}
            )
            
            if result.success:
                frontier_volatilities.append(result.fun)
                frontier_returns.append(target_return)
                frontier_weights.append(result.x)
        
        # Create DataFrame with results
        frontier_df = pd.DataFrame({
            'return': frontier_returns,
            'volatility': frontier_volatilities,
            'sharpe_ratio': [(r - self.risk_free_rate) / v 
                           for r, v in zip(frontier_returns, frontier_volatilities)]
        })
        
        return frontier_df