"""
Market Microstructure Data Handler for Portfolio Optimization
Designed for high-frequency data processing and order book analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import asyncio
import aiohttp
from datetime import datetime, timedelta

@dataclass
class OrderBookSnapshot:
    """Represents a full order book at a point in time"""
    timestamp: pd.Timestamp
    symbol: str
    bids: np.ndarray  # shape: (levels, 2) - [price, size]
    asks: np.ndarray  # shape: (levels, 2) - [price, size]
    
    @property
    def mid_price(self) -> float:
        """Calculate mid-price from best bid/ask"""
        if len(self.bids) > 0 and len(self.asks) > 0:
            return (self.bids[0, 0] + self.asks[0, 0]) / 2
        return np.nan
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if len(self.bids) > 0 and len(self.asks) > 0:
            return self.asks[0, 0] - self.bids[0, 0]
        return np.nan
    
    @property
    def weighted_mid_price(self) -> float:
        """Calculate size-weighted mid-price"""
        if len(self.bids) > 0 and len(self.asks) > 0:
            bid_size = self.bids[0, 1]
            ask_size = self.asks[0, 1]
            total_size = bid_size + ask_size
            if total_size > 0:
                return (self.bids[0, 0] * ask_size + self.asks[0, 0] * bid_size) / total_size
        return self.mid_price
    
    def get_depth_imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance using top N levels"""
        bid_volume = np.sum(self.bids[:levels, 1]) if len(self.bids) >= levels else 0
        ask_volume = np.sum(self.asks[:levels, 1]) if len(self.asks) >= levels else 0
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0.0

class MicrostructureDataHandler:
    """
    Handles high-frequency market data and microstructure analytics
    """
    
    def __init__(self, symbols: List[str], max_book_depth: int = 10):
        self.symbols = symbols
        self.max_book_depth = max_book_depth
        self.order_books: Dict[str, deque] = {symbol: deque(maxlen=10000) for symbol in symbols}
        self.trades: Dict[str, pd.DataFrame] = {}
        
    def process_order_book_update(self, symbol: str, timestamp: pd.Timestamp, 
                                 bids: List[Tuple[float, float]], 
                                 asks: List[Tuple[float, float]]) -> OrderBookSnapshot:
        """Process incoming order book data"""
        # Convert to numpy arrays for efficient computation
        bid_array = np.array(bids[:self.max_book_depth]) if bids else np.array([])
        ask_array = np.array(asks[:self.max_book_depth]) if asks else np.array([])
        
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bids=bid_array,
            asks=ask_array
        )
        
        self.order_books[symbol].append(snapshot)
        return snapshot
    
    def calculate_liquidity_metrics(self, symbol: str, 
                                  lookback_minutes: int = 5) -> Dict[str, float]:
        """Calculate various liquidity metrics from order book history"""
        if symbol not in self.order_books or len(self.order_books[symbol]) == 0:
            return {}
        
        # Get recent snapshots
        snapshots = list(self.order_books[symbol])
        current_time = snapshots[-1].timestamp
        cutoff_time = current_time - pd.Timedelta(minutes=lookback_minutes)
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {}
        
        # Calculate metrics
        spreads = [s.spread for s in recent_snapshots if not np.isnan(s.spread)]
        mid_prices = [s.mid_price for s in recent_snapshots if not np.isnan(s.mid_price)]
        imbalances = [s.get_depth_imbalance() for s in recent_snapshots]
        
        # Effective spread (time-weighted)
        effective_spread = np.mean(spreads) if spreads else np.nan
        
        # Realized volatility from mid-price changes
        if len(mid_prices) >= 2:
            returns = np.diff(np.log(mid_prices))
            realized_vol = np.std(returns) * np.sqrt(252 * 390 * 12)  # Annualized 5-min vol
        else:
            realized_vol = np.nan
        
        # Average depth at best bid/ask
        avg_bid_depth = np.mean([s.bids[0, 1] for s in recent_snapshots 
                                if len(s.bids) > 0])
        avg_ask_depth = np.mean([s.asks[0, 1] for s in recent_snapshots 
                                if len(s.asks) > 0])
        
        # Order book slope (price impact proxy)
        slopes = []
        for snapshot in recent_snapshots[-10:]:  # Use last 10 snapshots
            if len(snapshot.bids) >= 5 and len(snapshot.asks) >= 5:
                # Calculate average price impact per unit volume
                bid_prices = snapshot.bids[:5, 0]
                bid_volumes = snapshot.bids[:5, 1]
                bid_cumvol = np.cumsum(bid_volumes)
                
                ask_prices = snapshot.asks[:5, 0]
                ask_volumes = snapshot.asks[:5, 1]
                ask_cumvol = np.cumsum(ask_volumes)
                
                # Linear regression for price impact
                if len(bid_cumvol) > 1:
                    bid_slope = np.polyfit(bid_cumvol, bid_prices, 1)[0]
                    ask_slope = np.polyfit(ask_cumvol, ask_prices, 1)[0]
                    slopes.append((abs(bid_slope) + abs(ask_slope)) / 2)
        
        avg_slope = np.mean(slopes) if slopes else np.nan
        
        return {
            'effective_spread': effective_spread,
            'realized_volatility': realized_vol,
            'avg_bid_depth': avg_bid_depth,
            'avg_ask_depth': avg_ask_depth,
            'avg_imbalance': np.mean(imbalances),
            'order_book_slope': avg_slope,
            'spread_volatility': np.std(spreads) if len(spreads) > 1 else np.nan
        }
    
    def detect_liquidity_regimes(self, symbol: str) -> str:
        """Detect liquidity regime with realistic thresholds"""
        metrics = self.calculate_liquidity_metrics(symbol)
        if not metrics:
            return 'unknown'
        
        # REALISTIC thresholds for S&P 500 stocks
        # Effective spread in basis points
        spread_bps = metrics['effective_spread'] * 10000
        
        if spread_bps < 5:  # Less than 5 bps - ultra liquid
            return 'ultra_high_liquidity'
        elif spread_bps < 20:  # 5-20 bps - very liquid
            return 'high_liquidity'
        elif spread_bps < 100:  # 20-100 bps - normal liquidity
            return 'normal_liquidity'
        elif spread_bps < 200:  # 100-200 bps - moderate liquidity
            return 'moderate_liquidity'
        else:  # Over 200 bps - low liquidity
            return 'low_liquidity'
    
    def calculate_trade_metrics(self, symbol: str, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based microstructure metrics"""
        if trades_df.empty:
            return {}
        
        # Ensure proper datetime index
        if not isinstance(trades_df.index, pd.DatetimeIndex):
            trades_df.index = pd.to_datetime(trades_df['timestamp'])
        
        # Volume-weighted average price (VWAP)
        vwap = np.sum(trades_df['price'] * trades_df['volume']) / trades_df['volume'].sum()
        
        # Trade size distribution
        avg_trade_size = trades_df['volume'].mean()
        large_trade_ratio = (trades_df['volume'] > trades_df['volume'].quantile(0.9)).mean()
        
        # Kyle's lambda (price impact coefficient)
        # Regress price changes on signed volume
        if len(trades_df) >= 10:
            price_changes = trades_df['price'].pct_change().dropna()
            signed_volume = trades_df['volume'] * trades_df['side'].map({'buy': 1, 'sell': -1})
            signed_volume = signed_volume.iloc[1:]  # Align with price changes
            
            if len(price_changes) == len(signed_volume) and signed_volume.std() > 0:
                kyle_lambda = np.cov(price_changes, signed_volume)[0, 1] / np.var(signed_volume)
            else:
                kyle_lambda = np.nan
        else:
            kyle_lambda = np.nan
        
        # Probability of informed trading (PIN) proxy
        # Simplified version using trade imbalance
        buy_volume = trades_df[trades_df['side'] == 'buy']['volume'].sum()
        sell_volume = trades_df[trades_df['side'] == 'sell']['volume'].sum()
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            order_imbalance = abs(buy_volume - sell_volume) / total_volume
            pin_proxy = order_imbalance  # Simplified PIN proxy
        else:
            pin_proxy = np.nan
        
        return {
            'vwap': vwap,
            'avg_trade_size': avg_trade_size,
            'large_trade_ratio': large_trade_ratio,
            'kyle_lambda': kyle_lambda,
            'pin_proxy': pin_proxy
        }
    
    def simulate_market_impact(self, symbol: str, order_size: float, 
                             side: str = 'buy') -> Dict[str, float]:
        """
        Simulate market impact for a given order size
        Uses square-root model: Impact = spread/2 + lambda * sqrt(order_size/ADV)
        """
        if symbol not in self.order_books or len(self.order_books[symbol]) == 0:
            return {'error': 'No order book data available'}
        
        # Get current order book
        current_book = self.order_books[symbol][-1]
        
        # Calculate average daily volume (mock for now)
        adv = 1000000  # You would calculate this from historical data
        
        # Get spread
        spread = current_book.spread
        
        # Calculate temporary impact (square-root model)
        # Lambda typically ranges from 0.1 to 1.0 depending on the asset
        lambda_param = 0.3
        temp_impact = spread/2 + lambda_param * np.sqrt(order_size / adv)
        
        # Calculate permanent impact (usually 1/3 to 1/2 of temporary)
        perm_impact = temp_impact * 0.4
        
        # Walk the book to get actual impact
        if side == 'buy':
            book_side = current_book.asks
        else:
            book_side = current_book.bids
        
        remaining_size = order_size
        weighted_price = 0
        total_cost = 0
        
        for level in range(len(book_side)):
            level_price = book_side[level, 0]
            level_size = book_side[level, 1]
            
            fill_size = min(remaining_size, level_size)
            weighted_price += level_price * fill_size
            total_cost += level_price * fill_size
            remaining_size -= fill_size
            
            if remaining_size <= 0:
                break
        
        if remaining_size > 0:
            # Order exceeds visible book depth
            last_price = book_side[-1, 0] if len(book_side) > 0 else current_book.mid_price
            impact_price = last_price * (1 + temp_impact * (remaining_size / order_size))
            total_cost += impact_price * remaining_size
            weighted_price += impact_price * remaining_size
        
        avg_execution_price = total_cost / order_size
        realized_impact = abs(avg_execution_price - current_book.mid_price) / current_book.mid_price
        
        return {
            'mid_price': current_book.mid_price,
            'avg_execution_price': avg_execution_price,
            'temporary_impact_bps': temp_impact * 10000,
            'permanent_impact_bps': perm_impact * 10000,
            'realized_impact_bps': realized_impact * 10000,
            'total_cost': total_cost
        }

# Example usage for backtesting
def create_synthetic_order_book(
    mid_price: float,
    spread: float,
    depth_profile: str = 'normal',
    num_levels: int = 10,
    depth_multiplier: float = 1000
    ) -> OrderBookSnapshot:
    """Create synthetic order book with realistic parameters"""
    
    half_spread = spread / 2
    
    # Generate price levels
    bid_prices = mid_price - half_spread - np.arange(num_levels) * spread * 0.1
    ask_prices = mid_price + half_spread + np.arange(num_levels) * spread * 0.1
    
    # Generate sizes based on profile
    if depth_profile == 'deep':
        # Deep liquidity - lots of volume at each level
        bid_sizes = depth_multiplier * np.random.exponential(1, num_levels) * np.exp(-np.arange(num_levels) * 0.1)
        ask_sizes = depth_multiplier * np.random.exponential(1, num_levels) * np.exp(-np.arange(num_levels) * 0.1)
    elif depth_profile == 'normal':
        # Normal liquidity
        bid_sizes = depth_multiplier * 0.5 * np.random.exponential(1, num_levels) * np.exp(-np.arange(num_levels) * 0.2)
        ask_sizes = depth_multiplier * 0.5 * np.random.exponential(1, num_levels) * np.exp(-np.arange(num_levels) * 0.2)
    else:  # thin
        # Thin liquidity
        bid_sizes = depth_multiplier * 0.2 * np.random.exponential(1, num_levels) * np.exp(-np.arange(num_levels) * 0.3)
        ask_sizes = depth_multiplier * 0.2 * np.random.exponential(1, num_levels) * np.exp(-np.arange(num_levels) * 0.3)
    
    # Round sizes to integers
    bid_sizes = np.round(bid_sizes).astype(int)
    ask_sizes = np.round(ask_sizes).astype(int)
    
    # Ensure minimum size
    bid_sizes = np.maximum(bid_sizes, 100)
    ask_sizes = np.maximum(ask_sizes, 100)
    
    bids = np.column_stack((bid_prices, bid_sizes))
    asks = np.column_stack((ask_prices, ask_sizes))
    
    # Create OrderBookSnapshot with the correct parameters
    # The class should calculate mid_price and spread internally
    order_book = OrderBookSnapshot(
        timestamp=pd.Timestamp.now(),
        symbol='',
        bids=bids,
        asks=asks
    )
    
    # If OrderBookSnapshot doesn't calculate these automatically, set them as attributes
    #order_book.mid_price = mid_price
    #order_book.spread = spread
    
    return order_book