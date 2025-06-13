"""
Backtesting module for portfolio optimization.

This module provides tools for simulating portfolio strategies over historical
data with realistic trading constraints and costs.
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResults, RebalanceFrequency

__all__ = ['BacktestEngine', 'BacktestConfig', 'BacktestResults', 'RebalanceFrequency']