"""
Backtesting module for portfolio optimization.
"""

# Import all necessary components from engine
from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResults,
    RebalanceFrequency
)

# Make them available when importing from backtesting
__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResults',
    'RebalanceFrequency'
]