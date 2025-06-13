"""
Backtesting engine for portfolio strategies.

This module provides a realistic simulation framework for testing portfolio
strategies over historical data. It accounts for transaction costs, rebalancing
frequencies, and various real-world constraints that affect actual returns.

The key insight: Past performance doesn't guarantee future results, but
understanding how strategies behaved historically helps us make better decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """
    Defines how often the portfolio should be rebalanced.
    
    Think of this like deciding how often to tune up your car - too often
    and you waste money on maintenance, too rarely and performance degrades.
    """
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NEVER = "never"  # Buy and hold


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters.
    
    This class holds all the settings that control how realistic our backtest is.
    Each parameter represents a real-world friction or constraint that affects
    actual portfolio performance.
    """
    initial_capital: float = 100000  # Starting portfolio value
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    transaction_cost_pct: float = 0.001  # 0.1% per trade (10 basis points)
    slippage_pct: float = 0.0005  # 0.05% price impact
    min_position_size: float = 0.01  # Minimum 1% position
    max_position_size: float = 1.0  # Maximum 100% in one asset
    
    # Tax considerations (simplified)
    tax_rate_short_term: float = 0.35  # Short-term capital gains tax
    tax_rate_long_term: float = 0.15  # Long-term capital gains tax
    
    # Risk constraints
    max_leverage: float = 1.0  # No leverage by default
    stop_loss_pct: Optional[float] = None  # Optional stop-loss
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.transaction_cost_pct < 0 or self.transaction_cost_pct > 0.01:
            raise ValueError("Transaction costs should be between 0 and 1%")
        if self.min_position_size < 0 or self.min_position_size > self.max_position_size:
            raise ValueError("Invalid position size constraints")


@dataclass
class BacktestResults:
    """
    Comprehensive results from a backtest simulation.
    
    This class contains everything you need to understand how a strategy
    performed, not just the returns but also risk metrics, costs, and
    detailed analytics about the portfolio's behavior over time.
    """
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Cost analysis
    total_transaction_costs: float
    total_slippage_costs: float
    turnover_rate: float  # How often positions change
    
    # Time series data
    portfolio_values: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    
    # Risk metrics over time
    rolling_volatility: pd.Series
    drawdown_series: pd.Series
    
    # Summary statistics
    win_rate: float  # Percentage of positive return periods
    avg_win: float  # Average positive return
    avg_loss: float  # Average negative return
    best_period: float
    worst_period: float
    
    def summary_report(self) -> str:
        """Generate a human-readable summary of backtest results."""
        return f"""
Backtest Performance Summary
============================

Returns:
--------
Total Return: {self.total_return:.1%}
Annualized Return: {self.annualized_return:.1%}
Volatility: {self.volatility:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Maximum Drawdown: {self.max_drawdown:.1%}

Trading Costs:
--------------
Transaction Costs: ${self.total_transaction_costs:,.2f}
Slippage Costs: ${self.total_slippage_costs:,.2f}
Portfolio Turnover: {self.turnover_rate:.1%} annually

Risk Profile:
-------------
Win Rate: {self.win_rate:.1%}
Average Win: {self.avg_win:.2%}
Average Loss: {self.avg_loss:.2%}
Best Period: {self.best_period:.2%}
Worst Period: {self.worst_period:.2%}

Final Portfolio Value: ${self.portfolio_values.iloc[-1]:,.2f}
"""


class BacktestEngine:
    """
    Main backtesting engine for portfolio strategies.
    
    This engine simulates the entire lifecycle of a portfolio strategy:
    1. Starting with initial capital
    2. Generating trading signals based on the strategy
    3. Executing trades with realistic costs
    4. Tracking performance over time
    5. Calculating comprehensive metrics
    
    The goal is to answer: "How would this strategy have actually performed?"
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Backtesting configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        
    def run_backtest(
        self,
        prices: pd.DataFrame,
        strategy: Callable,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResults:
        """
        Run a complete backtest simulation.
        
        This is the main method that orchestrates the entire backtesting process.
        It takes historical prices and a strategy function, then simulates trading
        over the specified period.
        
        Args:
            prices: DataFrame of historical prices (assets as columns)
            strategy: Function that returns target weights given prices up to that point
            start_date: Start of backtest period (uses first date if None)
            end_date: End of backtest period (uses last date if None)
            
        Returns:
            BacktestResults object with comprehensive performance metrics
        """
        # Prepare data for the backtest period
        prices = self._prepare_data(prices, start_date, end_date)
        
        # Initialize tracking variables
        dates = prices.index
        n_assets = len(prices.columns)
        n_periods = len(dates)
        
        # Storage for results
        portfolio_values = np.zeros(n_periods)
        positions = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        trades = []
        costs = {'transaction': 0, 'slippage': 0}
        
        # Initialize portfolio
        current_weights = np.zeros(n_assets)
        current_cash = self.config.initial_capital
        current_positions = np.zeros(n_assets)  # Number of shares
        
        # Determine rebalancing dates
        rebalance_dates = self._get_rebalance_dates(dates)
        
        logger.info(f"Starting backtest from {dates[0]} to {dates[-1]}")
        logger.info(f"Rebalancing frequency: {self.config.rebalance_frequency.value}")
        logger.info(f"Number of rebalance dates: {len(rebalance_dates)}")
        
        # Main backtest loop
        for i, date in enumerate(dates):
            current_prices = prices.loc[date].values
            
            # Calculate current portfolio value
            position_values = current_positions * current_prices
            portfolio_value = position_values.sum() + current_cash
            portfolio_values[i] = portfolio_value
            
            # Record current positions
            positions.loc[date] = current_positions
            
            # Check if we need to rebalance
            if date in rebalance_dates and i > 0:  # Don't rebalance on first day
                # Get historical data up to this point
                historical_prices = prices.loc[:date]
                
                # Generate target weights using the strategy
                target_weights = strategy(historical_prices)
                
                # Ensure weights are valid
                target_weights = self._validate_weights(target_weights)
                
                # Calculate trades needed
                target_positions = (portfolio_value * target_weights) / current_prices
                trades_needed = target_positions - current_positions
                
                # Execute trades with costs
                trade_costs = self._execute_trades(
                    trades_needed, 
                    current_prices, 
                    portfolio_value,
                    date
                )
                
                # Update positions and cash
                current_positions = target_positions
                costs['transaction'] += trade_costs['transaction']
                costs['slippage'] += trade_costs['slippage']
                current_cash -= trade_costs['total']
                
                # Record trades
                for j, (asset, trade_size) in enumerate(zip(prices.columns, trades_needed)):
                    if abs(trade_size) > 0.01:  # Only record meaningful trades
                        trades.append({
                            'date': date,
                            'asset': asset,
                            'quantity': trade_size,
                            'price': current_prices[j],
                            'value': trade_size * current_prices[j],
                            'cost': trade_costs['per_asset'][j]
                        })
        
        # Calculate final metrics
        results = self._calculate_results(
            portfolio_values,
            positions,
            pd.DataFrame(trades) if trades else pd.DataFrame(),
            costs,
            prices
        )
        
        return results
    
    def _prepare_data(
        self, 
        prices: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Prepare and validate price data for backtesting.
        
        This method ensures we have clean, properly formatted data with no gaps
        or anomalies that could corrupt our backtest results.
        
        Updated for pandas 2.0+ compatibility.
        """
        # Handle date filtering
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
            
        # Forward fill missing values (assumes holding through weekends/holidays)
        # Using the new pandas syntax
        prices = prices.ffill()
        
        # Drop any remaining NaN values
        prices = prices.dropna()
        
        if len(prices) < 20:
            raise ValueError("Insufficient data for backtesting (need at least 20 periods)")
            
        return prices
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[datetime]:
        """
        Determine when to rebalance based on the frequency setting.
        
        This is like setting a schedule for portfolio maintenance - some investors
        prefer frequent adjustments while others prefer to let positions ride.
        
        Updated for pandas 2.0+ compatibility with improved date handling.
        """
        if self.config.rebalance_frequency == RebalanceFrequency.NEVER:
            return [dates[0]]  # Only initial allocation
            
        elif self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            return dates.tolist()
            
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            # First trading day of each week
            return dates[dates.weekday == 0].tolist()
            
        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            # First trading day of each month
            # Create a series to make operations cleaner
            date_series = pd.Series(dates, index=dates)
            
            # Find where the month changes
            month_changes = date_series.dt.month.diff() != 0
            month_changes.iloc[0] = True  # Include first date
            
            return dates[month_changes].tolist()
            
        elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            # First trading day of each quarter
            date_series = pd.Series(dates, index=dates)
            
            # Check if month is start of quarter (1, 4, 7, 10)
            is_quarter_month = dates.month.isin([1, 4, 7, 10])
            
            # Find where the month changes
            month_changes = date_series.dt.month.diff() != 0
            month_changes.iloc[0] = True  # Include first date
            
            # Combine conditions: must be quarter month AND month change
            quarterly_dates = dates[is_quarter_month & month_changes]
            
            # Make sure we include the first date if it's not already included
            if len(quarterly_dates) > 0 and quarterly_dates[0] != dates[0]:
                quarterly_dates = dates[:1].append(quarterly_dates)
                
            return quarterly_dates.tolist()
            
        elif self.config.rebalance_frequency == RebalanceFrequency.YEARLY:
            # First trading day of each year
            date_series = pd.Series(dates, index=dates)
            
            # Find where the year changes
            year_changes = date_series.dt.year.diff() != 0
            year_changes.iloc[0] = True  # Include first date
            
            return dates[year_changes].tolist()
    
    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Ensure portfolio weights meet all constraints.
        
        This method enforces real-world constraints like position limits and
        ensures the portfolio is fully invested (weights sum to 1).
        """
        # Clip weights to position size limits
        weights = np.clip(weights, self.config.min_position_size, 
                         self.config.max_position_size)
        
        # Renormalize to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Equal weight if all weights are zero
            weights = np.ones_like(weights) / len(weights)
            
        return weights
    
    def _execute_trades(
        self,
        trades_needed: np.ndarray,
        current_prices: np.ndarray,
        portfolio_value: float,
        date: datetime
    ) -> Dict[str, float]:
        """
        Simulate trade execution with realistic costs.
        
        This method models the real costs of trading, including both explicit
        costs (commissions) and implicit costs (market impact/slippage).
        """
        trade_values = np.abs(trades_needed * current_prices)
        
        # Calculate transaction costs (percentage of trade value)
        transaction_costs = trade_values * self.config.transaction_cost_pct
        
        # Calculate slippage (market impact)
        # Slippage is higher for larger trades
        trade_pct_of_portfolio = trade_values / portfolio_value
        slippage_multiplier = 1 + trade_pct_of_portfolio  # Larger trades have more impact
        slippage_costs = trade_values * self.config.slippage_pct * slippage_multiplier
        
        total_costs = transaction_costs + slippage_costs
        
        return {
            'transaction': transaction_costs.sum(),
            'slippage': slippage_costs.sum(),
            'total': total_costs.sum(),
            'per_asset': total_costs
        }
    
    def _calculate_results(
        self,
        portfolio_values: np.ndarray,
        positions: pd.DataFrame,
        trades: pd.DataFrame,
        costs: Dict[str, float],
        prices: pd.DataFrame
    ) -> BacktestResults:
        """
        Calculate comprehensive performance metrics from backtest data.
        
        This method computes all the statistics that help us understand not just
        how much money the strategy made, but how it made it and what risks
        were taken along the way.
        """
        # Convert portfolio values to series
        portfolio_series = pd.Series(portfolio_values, index=positions.index)
        
        # Calculate returns
        returns = portfolio_series.pct_change().dropna()
        
        # Basic performance metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        years = len(portfolio_values) / 252  # Approximate trading days per year
        annualized_return = (1 + total_return) ** (1 / years) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate from config)
        excess_returns = returns - self.config.transaction_cost_pct / 252
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown_series = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown_series.min()
        
        # Win/loss analysis
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        # Turnover calculation
        if not trades.empty:
            # Group trades by date and sum absolute values
            daily_turnover = trades.groupby('date')['value'].apply(lambda x: x.abs().sum())
            
            # Calculate average portfolio value
            avg_portfolio_value = portfolio_series.mean()
            
            # Annualized turnover rate
            total_turnover = daily_turnover.sum()
            turnover_rate = (total_turnover / avg_portfolio_value) / years
        else:
            turnover_rate = 0
        
        # Rolling metrics
        rolling_window = min(63, len(returns) // 4)  # ~3 months or quarter of data
        rolling_volatility = returns.rolling(rolling_window).std() * np.sqrt(252)
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_transaction_costs=costs['transaction'],
            total_slippage_costs=costs['slippage'],
            turnover_rate=turnover_rate,
            portfolio_values=portfolio_series,
            returns=returns,
            positions=positions,
            trades=trades,
            rolling_volatility=rolling_volatility,
            drawdown_series=drawdown_series,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_period=returns.max() if len(returns) > 0 else 0,
            worst_period=returns.min() if len(returns) > 0 else 0
        )