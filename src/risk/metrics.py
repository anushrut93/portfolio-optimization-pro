"""
Risk metrics module for portfolio analysis.

This module provides comprehensive risk measurement tools used by professional
portfolio managers to understand and quantify various types of investment risk.
Each metric captures a different aspect of risk, providing a complete picture
of potential portfolio losses.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Optional
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """
    Container for comprehensive risk metrics of a portfolio.
    
    This class holds all calculated risk metrics in one place, making it easy
    to access and display risk information. Think of it as a medical report
    for your portfolio's health.
    """
    volatility: float              # Standard deviation of returns
    downside_deviation: float      # Volatility of negative returns only
    var_95: float                 # Value at Risk at 95% confidence
    var_99: float                 # Value at Risk at 99% confidence
    cvar_95: float                # Conditional VaR at 95% confidence
    cvar_99: float                # Conditional VaR at 99% confidence
    max_drawdown: float           # Maximum peak-to-trough decline
    max_drawdown_duration: int    # Days to recover from max drawdown
    sharpe_ratio: float           # Risk-adjusted return metric
    sortino_ratio: float          # Downside risk-adjusted return metric
    calmar_ratio: float           # Return to max drawdown ratio
    
    def __str__(self) -> str:
        """Create a formatted report of all risk metrics."""
        return f"""
Portfolio Risk Analysis Report
==============================

Volatility Measures:
-------------------
Annual Volatility: {self.volatility:.1%}
Downside Deviation: {self.downside_deviation:.1%}

Value at Risk (VaR):
-------------------
95% VaR: {self.var_95:.1%} (1 in 20 days)
99% VaR: {self.var_99:.1%} (1 in 100 days)

Tail Risk (CVaR/Expected Shortfall):
-----------------------------------
95% CVaR: {self.cvar_95:.1%}
99% CVaR: {self.cvar_99:.1%}

Drawdown Analysis:
-----------------
Maximum Drawdown: {self.max_drawdown:.1%}
Recovery Period: {self.max_drawdown_duration} days

Risk-Adjusted Performance:
-------------------------
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Calmar Ratio: {self.calmar_ratio:.2f}
"""


class RiskAnalyzer:
    """
    Comprehensive risk analysis for portfolios.
    
    This class implements various risk metrics used in professional portfolio
    management. It can analyze both individual assets and portfolios,
    providing insights into different dimensions of risk.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the risk analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate returns from price series.
        
        We use logarithmic returns because they have better statistical
        properties for risk analysis (they're additive over time).
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns
    
    def calculate_portfolio_returns(
        self, 
        prices: pd.DataFrame, 
        weights: np.ndarray
    ) -> pd.Series:
        """
        Calculate weighted portfolio returns from individual asset prices.
        
        This method handles the complexity of combining multiple assets into
        a single portfolio return series, accounting for the weight of each asset.
        
        Args:
            prices: DataFrame with asset prices (assets as columns)
            weights: Array of portfolio weights (must sum to 1)
            
        Returns:
            Series of portfolio returns
        """
        # First, calculate returns for each asset
        asset_returns = prices.pct_change().dropna()
        
        # Then calculate weighted portfolio returns
        # This is like calculating a weighted average grade across different subjects
        portfolio_returns = (asset_returns * weights).sum(axis=1)
        
        return portfolio_returns
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level.
        
        VaR tells us the maximum expected loss over a given time period
        at a specified confidence level. For example, 95% VaR of -2% means
        we expect to lose no more than 2% on 95% of days.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR as a negative percentage
        """
        if method == 'historical':
            # Historical VaR: simply find the percentile in actual returns
            # This method makes no assumptions about return distribution
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Parametric VaR: assumes returns follow normal distribution
            # Faster but less accurate for non-normal returns
            mean = returns.mean()
            std = returns.std()
            var = mean + std * stats.norm.ppf(1 - confidence_level)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to annual if using daily returns
        if len(returns) > 252:  # More than a year of daily data
            var = var * np.sqrt(252)  # Annualize
            
        return var
    
    def calculate_cvar(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.
        
        CVaR measures the average loss when losses exceed the VaR threshold.
        It answers: "When things go bad, how bad do they typically get?"
        
        This is considered a better risk measure than VaR because it captures
        the severity of losses in the tail of the distribution.
        """
        var = self.calculate_var(returns, confidence_level, method='historical')
        
        # Find all returns worse than VaR
        tail_losses = returns[returns <= var]
        
        if len(tail_losses) == 0:
            return var  # If no losses beyond VaR, CVaR equals VaR
        
        cvar = tail_losses.mean()
        
        # Annualize if using daily returns
        if len(returns) > 252:
            cvar = cvar * np.sqrt(252)
            
        return cvar
    
    def calculate_max_drawdown(
        self, 
        prices: pd.Series
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and recovery time.
        
        Maximum drawdown is the largest peak-to-trough decline in value.
        This is often the most psychologically important risk metric because
        it represents the worst experience an investor would have faced.
        
        Returns:
            Tuple of (max_drawdown, recovery_duration_in_days)
        """
        # Calculate cumulative returns to track portfolio value
        cumulative = (1 + self.calculate_returns(prices)).cumprod()
        
        # Calculate running maximum (highest point reached so far)
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown at each point (current value vs peak)
        drawdown = (cumulative - running_max) / running_max
        
        # Find the maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find when max drawdown occurred
        max_dd_idx = drawdown.idxmin()
        
        # Calculate recovery time
        # First, find the peak before the drawdown
        peak_idx = cumulative[:max_dd_idx].idxmax()
        
        # Then find recovery point (if it exists)
        peak_value = cumulative[peak_idx]
        recovery_data = cumulative[max_dd_idx:]
        
        recovery_points = recovery_data[recovery_data >= peak_value]
        
        if len(recovery_points) > 0:
            recovery_idx = recovery_points.index[0]
            recovery_duration = len(cumulative[max_dd_idx:recovery_idx])
        else:
            # Haven't recovered yet
            recovery_duration = len(cumulative[max_dd_idx:])
            
        return max_drawdown, recovery_duration
    
    def calculate_downside_deviation(
        self, 
        returns: pd.Series, 
        target_return: float = 0
    ) -> float:
        """
        Calculate downside deviation (semi-deviation).
        
        This measures volatility of returns below a target (usually 0).
        Investors typically don't mind upside volatility - they only care
        about downside risk. This metric captures that asymmetry.
        """
        # Filter returns below target
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate standard deviation of downside returns
        downside_dev = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        # Annualize if using daily returns
        if len(returns) > 252:
            downside_dev = downside_dev * np.sqrt(252)
            
        return downside_dev
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        target_return: float = 0
    ) -> float:
        """
        Calculate Sortino ratio - a variation of Sharpe that uses downside deviation.
        
        This is often considered superior to Sharpe ratio because it doesn't
        penalize upside volatility. A fund that occasionally has spectacular
        gains shouldn't be penalized for that volatility.
        """
        excess_returns = returns.mean() - self.risk_free_rate / 252  # Daily excess
        downside_dev = self.calculate_downside_deviation(returns, target_return)
        
        if downside_dev == 0:
            return np.inf if excess_returns > 0 else 0
            
        # Annualize
        sortino = (excess_returns * 252) / downside_dev
        
        return sortino
    
    def analyze_risk(
        self, 
        prices: Union[pd.Series, pd.DataFrame],
        weights: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """
        Perform comprehensive risk analysis on asset prices or portfolio.
        
        This is the main method that combines all risk metrics into a single
        analysis. It handles both individual assets (Series) and portfolios
        (DataFrame with weights).
        
        Args:
            prices: Series for single asset or DataFrame for multiple assets
            weights: Portfolio weights (required if prices is DataFrame)
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        # Handle portfolio vs single asset
        if isinstance(prices, pd.DataFrame):
            if weights is None:
                raise ValueError("Weights required for portfolio analysis")
            returns = self.calculate_portfolio_returns(prices, weights)
            price_series = (prices * weights).sum(axis=1)
        else:
            returns = self.calculate_returns(prices)
            price_series = prices
            
        # Calculate all metrics
        logger.info("Calculating comprehensive risk metrics...")
        
        # Basic metrics
        volatility = returns.std() * np.sqrt(252)
        annual_return = returns.mean() * 252
        
        # VaR and CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # Drawdown analysis
        max_dd, dd_duration = self.calculate_max_drawdown(price_series)
        
        # Downside risk
        downside_dev = self.calculate_downside_deviation(returns)
        
        # Risk-adjusted ratios
        sharpe = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino = self.calculate_sortino_ratio(returns)
        calmar = -annual_return / max_dd if max_dd < 0 else 0
        
        return RiskMetrics(
            volatility=volatility,
            downside_deviation=downside_dev,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar
        )