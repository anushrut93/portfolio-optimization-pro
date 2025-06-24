"""
Enhanced Backtesting Engine for Long/Short Strategies
Standalone version with minimal dependencies on existing code
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Basic backtest configuration"""
    initial_capital: float = 1_000_000
    rebalance_frequency: str = 'monthly'
    transaction_cost_bps: float = 10
    slippage_bps: float = 5
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class LongShortBacktestConfig(BacktestConfig):
    """Extended configuration for long/short backtesting"""
    # Short-specific costs
    borrow_rates: Optional[pd.Series] = None  # Annual borrow rates per stock
    locate_fees: Optional[pd.Series] = None   # One-time locate fees
    forced_buyin_rate: float = 0.02          # Annual forced buy-in probability
    
    # Long/short constraints
    max_gross_leverage: float = 2.0          # Maximum gross exposure
    max_net_exposure: float = 0.3            # Maximum net long/short
    min_short_position: float = -0.10        # Maximum short per position
    
    # Risk limits
    max_sector_exposure: float = 0.20        # Maximum net sector exposure
    max_beta: float = 0.2                    # Maximum absolute beta to market


@dataclass
class BacktestResult:
    """Container for backtest results"""
    portfolio_value: pd.Series
    returns: pd.Series
    positions_history: List[pd.Series]
    metrics: Dict[str, float]
    
    
@dataclass
class LongShortBacktestResult:
    """Container for long/short backtest results"""
    base_result: BacktestResult
    long_short_metrics: Dict
    total_borrowing_cost: float
    adjusted_returns: pd.Series
    exposure_history: pd.DataFrame
    
    def plot_exposures(self):
        """Plot gross and net exposures over time"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Gross and net exposure
        self.exposure_history.set_index('date')[['gross_leverage', 'net_exposure']].plot(ax=ax1)
        ax1.set_ylabel('Exposure')
        ax1.set_title('Portfolio Exposures Over Time')
        ax1.legend(['Gross', 'Net'])
        ax1.grid(True, alpha=0.3)
        
        # Long vs short exposure
        self.exposure_history.set_index('date')[['long_exposure', 'short_exposure']].abs().plot(ax=ax2)
        ax2.set_ylabel('Exposure (Absolute)')
        ax2.set_title('Long vs Short Exposure')
        ax2.legend(['Long', 'Short'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class BacktestEngine:
    """Basic backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_value_history = []
        self.positions_history = []
        self.returns_history = []
        
    def run_backtest(self, strategy, price_data: pd.DataFrame, 
                    start_date: str, end_date: str) -> BacktestResult:
        """Run a simple backtest"""
        
        # Filter data to date range
        mask = (price_data.index >= start_date) & (price_data.index <= end_date)
        price_data = price_data.loc[mask]
        
        # Initialize
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        portfolio_values = [self.config.initial_capital]
        all_returns = []
        
        # Simple monthly rebalancing
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]
            
            # Get data up to current date
            historical_data = price_data[:current_date]
            
            # Generate signals
            signals = strategy.generate_signals(historical_data)
            
            # Get positions
            positions = strategy.construct_portfolio(signals)
            self.positions_history.append(positions)
            
            # Calculate returns for the period
            period_prices = price_data[current_date:next_date]
            if len(period_prices) > 1:
                period_returns = period_prices.pct_change().dropna()
                
                # Portfolio returns
                portfolio_returns = (period_returns * positions).sum(axis=1)
                
                # Apply costs
                turnover = positions.sub(self.positions_history[-2] if i > 0 else 0).abs().sum()
                cost = turnover * self.config.transaction_cost_bps / 10000
                portfolio_returns.iloc[0] -= cost
                
                # Update portfolio value
                for ret in portfolio_returns:
                    portfolio_values.append(portfolio_values[-1] * (1 + ret))
                    all_returns.append(ret)
        
        # Create results
        portfolio_series = pd.Series(portfolio_values[:len(dates)], index=dates[:len(portfolio_values)])
        returns_series = portfolio_series.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns_series)
        
        return BacktestResult(
            portfolio_value=portfolio_series,
            returns=returns_series,
            positions_history=self.positions_history,
            metrics=metrics
        )
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


class LongShortBacktestEngine(BacktestEngine):
    """
    Specialized backtesting engine for long/short strategies
    """
    
    def __init__(self, config: LongShortBacktestConfig):
        super().__init__(config)
        self.ls_config = config
        self.short_positions_history = []
        self.borrowing_costs_history = []
        self.exposure_history = []
        self.returns_history = []
        
    def run_backtest(self, 
                    strategy,
                    price_data: pd.DataFrame,
                    start_date: str,
                    end_date: str,
                    ml_predictions: Optional[Dict] = None) -> LongShortBacktestResult:
        """
        Run long/short backtest with full cost accounting
        """
        # Get base backtest results
        base_result = super().run_backtest(strategy, price_data, start_date, end_date)
        
        # Calculate long/short specific metrics
        ls_metrics = self._calculate_long_short_metrics(base_result)
        
        # Calculate total borrowing costs
        total_borrow_cost = self._calculate_total_borrowing_costs()
        
        # Adjust returns for borrowing costs
        adjusted_returns = base_result.returns - total_borrow_cost / self.config.initial_capital
        
        # Create exposure history DataFrame
        exposure_df = pd.DataFrame(self.exposure_history) if self.exposure_history else pd.DataFrame()
        
        return LongShortBacktestResult(
            base_result=base_result,
            long_short_metrics=ls_metrics,
            total_borrowing_cost=total_borrow_cost,
            adjusted_returns=adjusted_returns,
            exposure_history=exposure_df
        )
    
    def _calculate_portfolio_returns(self, 
                                   weights: pd.Series,
                                   returns: pd.DataFrame,
                                   transaction_costs: float) -> pd.Series:
        """
        Override to handle short positions correctly
        """
        # Base portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Add borrowing costs for shorts
        if self.ls_config.borrow_rates is not None:
            short_positions = weights[weights < 0]
            if len(short_positions) > 0:
                # Daily borrowing cost
                daily_borrow_cost = (
                    short_positions.abs() * 
                    self.ls_config.borrow_rates[short_positions.index] / 
                    252 / 10000  # Convert annual bps to daily decimal
                ).sum()
                
                portfolio_returns = portfolio_returns - daily_borrow_cost
                
                # Track borrowing costs
                self.borrowing_costs_history.append({
                    'date': returns.index[-1],
                    'daily_cost': daily_borrow_cost,
                    'annual_cost_bps': daily_borrow_cost * 252 * 10000
                })
        
        # Track positions
        self.short_positions_history.append({
            'date': returns.index[-1] if len(returns) > 0 else pd.Timestamp.now(),
            'short_positions': weights[weights < 0].to_dict(),
            'n_shorts': len(weights[weights < 0]),
            'total_short_exposure': weights[weights < 0].sum()
        })
        
        # Track exposures
        self.exposure_history.append({
            'date': pd.Timestamp.now(),
            'gross_leverage': weights.abs().sum(),
            'net_exposure': weights.sum(),
            'long_exposure': weights[weights > 0].sum(),
            'short_exposure': weights[weights < 0].sum(),
            'n_long': len(weights[weights > 0]),
            'n_short': len(weights[weights < 0])
        })
        
        # Store for attribution
        self.returns_history.append({
            'weights': weights,
            'returns': returns.iloc[-1] if len(returns) > 0 else pd.Series()
        })
        
        # Apply transaction costs
        portfolio_returns.iloc[0] -= transaction_costs
        
        return portfolio_returns
    
    def _calculate_long_short_metrics(self, base_result) -> Dict:
        """Calculate long/short specific performance metrics"""
        
        if not self.exposure_history:
            return {}
            
        exposure_df = pd.DataFrame(self.exposure_history)
        
        metrics = {
            # Exposure metrics
            'avg_gross_leverage': exposure_df['gross_leverage'].mean() if len(exposure_df) > 0 else 0,
            'avg_net_exposure': exposure_df['net_exposure'].mean() if len(exposure_df) > 0 else 0,
            'avg_long_exposure': exposure_df['long_exposure'].mean() if len(exposure_df) > 0 else 0,
            'avg_short_exposure': exposure_df['short_exposure'].mean() if len(exposure_df) > 0 else 0,
            
            # Position metrics
            'avg_n_long': exposure_df['n_long'].mean() if len(exposure_df) > 0 else 0,
            'avg_n_short': exposure_df['n_short'].mean() if len(exposure_df) > 0 else 0,
            'max_gross_leverage': exposure_df['gross_leverage'].max() if len(exposure_df) > 0 else 0,
            'min_net_exposure': exposure_df['net_exposure'].min() if len(exposure_df) > 0 else 0,
            'max_net_exposure': exposure_df['net_exposure'].max() if len(exposure_df) > 0 else 0,
            
            # Attribution (simplified)
            'long_contribution': 0.0,  # Would calculate if we had the data
            'short_contribution': 0.0,
            'long_short_correlation': 0.0,
            'exposure_stability': exposure_df['net_exposure'].std() if len(exposure_df) > 1 else 0,
        }
        
        return metrics
    
    def _calculate_total_borrowing_costs(self) -> float:
        """Calculate total borrowing costs over backtest period"""
        if not self.borrowing_costs_history:
            return 0.0
            
        total_cost = sum(cost['daily_cost'] for cost in self.borrowing_costs_history)
        return total_cost