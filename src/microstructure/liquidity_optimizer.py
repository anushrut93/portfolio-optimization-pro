"""
Liquidity-Aware Portfolio Optimization
Integrates market microstructure constraints and liquidity risk into portfolio construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp
from dataclasses import dataclass

@dataclass
class LiquidityConstraints:
    """Liquidity-based constraints for portfolio optimization"""
    max_participation_rate: float = 0.05  # Max % of daily volume
    max_spread_cost: float = 0.001        # Max acceptable spread cost (10 bps)
    min_days_to_liquidate: float = 1.0    # Minimum liquidation time
    max_market_impact: float = 0.005      # Max acceptable market impact (50 bps)

class LiquidityAwareOptimizer:
    """
    Portfolio optimizer that incorporates liquidity constraints and market microstructure
    """
    
    def __init__(self, 
                 expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 liquidity_metrics: Dict[str, Dict[str, float]],
                 impact_model: 'MarketImpactModel',
                 constraints: LiquidityConstraints = None):
        """
        Initialize with returns, risk, and liquidity data
        
        liquidity_metrics should contain for each asset:
        - adv: average daily volume
        - spread: average bid-ask spread
        - volatility: return volatility
        - depth: average market depth
        - price: current price
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.liquidity_metrics = liquidity_metrics
        self.impact_model = impact_model
        self.constraints = constraints or LiquidityConstraints()
        self.n_assets = len(expected_returns)
        
    def calculate_liquidity_score(self, symbol: str) -> float:
        """
        Calculate composite liquidity score for an asset (0-1, higher is better)
        """
        metrics = self.liquidity_metrics.get(symbol, {})
        
        # Normalize each component
        spread_score = 1 / (1 + metrics.get('spread', 0.01) * 1000)  # Lower spread = higher score
        volume_score = min(metrics.get('adv', 0) / 1e6, 1)  # Normalize by 1M shares
        depth_score = min(metrics.get('depth', 0) / 1000, 1)  # Normalize by 1000 shares
        
        # Composite score (weighted average)
        liquidity_score = 0.4 * spread_score + 0.4 * volume_score + 0.2 * depth_score
        
        return liquidity_score
    
    def calculate_portfolio_liquidity(self, weights: pd.Series, 
                                    portfolio_value: float) -> Dict[str, float]:
        """
        Calculate portfolio-level liquidity metrics
        """
        position_values = weights * portfolio_value
        
        # Time to liquidate each position
        liquidation_times = {}
        total_liquidation_time = 0
        
        for symbol, position in position_values.items():
            if abs(position) > 0:
                adv_dollars = self.liquidity_metrics[symbol]['adv'] * \
                            self.liquidity_metrics[symbol].get('price', 100)
                # Assume we can trade at max participation rate
                daily_capacity = adv_dollars * self.constraints.max_participation_rate
                days_to_liquidate = abs(position) / daily_capacity
                liquidation_times[symbol] = days_to_liquidate
                total_liquidation_time = max(total_liquidation_time, days_to_liquidate)
        
        # Portfolio liquidity score (weighted by position size)
        portfolio_liquidity_score = 0
        for symbol, weight in weights.items():
            if weight > 0:
                portfolio_liquidity_score += weight * self.calculate_liquidity_score(symbol)
        
        # Concentration in illiquid assets
        illiquid_concentration = sum(weights[s] for s in weights.index 
                                   if self.calculate_liquidity_score(s) < 0.3)
        
        return {
            'portfolio_liquidity_score': portfolio_liquidity_score,
            'max_liquidation_time': total_liquidation_time,
            'avg_liquidation_time': np.mean(list(liquidation_times.values())) if liquidation_times else 0,
            'illiquid_concentration': illiquid_concentration,
            'position_liquidation_times': liquidation_times
        }
    
    def optimize_with_liquidity_constraints(self, 
                                          portfolio_value: float,
                                          current_weights: Optional[pd.Series] = None,
                                          method: str = 'cvxpy') -> Dict[str, any]:
        """
        Optimize portfolio with liquidity constraints
        Uses CVXPY for convex optimization with complex constraints
        """
        if method == 'cvxpy':
            return self._optimize_cvxpy(portfolio_value, current_weights)
        else:
            return self._optimize_scipy(portfolio_value, current_weights)
    
    def _optimize_cvxpy(self, portfolio_value: float, 
                       current_weights: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Optimize using CVXPY (more robust for complex constraints)
        """
        n = self.n_assets
        
        # Decision variables
        w = cp.Variable(n)
        
        # Expected return
        ret = self.expected_returns.values @ w
        
        # Risk (portfolio variance)
        risk = cp.quad_form(w, self.cov_matrix.values)
        
        # Transaction costs (if we have current weights)
        if current_weights is not None:
            trades = w - current_weights.values
            
            # Approximate transaction costs using L1 norm
            # In practice, you'd use the actual impact model here
            transaction_cost = cp.sum(cp.abs(trades)) * 0.003  # 10 bps per trade
        else:
            transaction_cost = 0
        
        # Objective: maximize return - risk penalty - transaction costs
        objective = cp.Maximize(ret - 0.5 * risk - transaction_cost)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= 0,          # Long only
            w <= 0.3         # Max position size 30%
        ]
        
        # Liquidity constraints
        for i, symbol in enumerate(self.expected_returns.index):
            metrics = self.liquidity_metrics[symbol]
            adv_dollars = metrics['adv'] * metrics.get('price', 100)
            
            # Position size constraint based on ADV
            max_position = adv_dollars * self.constraints.max_participation_rate * \
                         self.constraints.min_days_to_liquidate
            constraints.append(w[i] * portfolio_value <= max_position)
            
            # Penalize illiquid positions
            liquidity_score = self.calculate_liquidity_score(symbol)
            if liquidity_score < 0.3:  # Illiquid threshold
                constraints.append(w[i] <= 0.05)  # Max 5% in illiquid assets
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed: {problem.status}")
        
        # Extract results
        optimal_weights = pd.Series(w.value, index=self.expected_returns.index)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(self.cov_matrix, optimal_weights)))
        liquidity_metrics = self.calculate_portfolio_liquidity(optimal_weights, portfolio_value)
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': (portfolio_return - 0.02) / portfolio_risk,  # Assuming 2% risk-free rate
            'liquidity_metrics': liquidity_metrics,
            'optimization_status': problem.status
        }
    
    def _optimize_scipy(self, portfolio_value: float, 
                       current_weights: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Fallback optimization using scipy (if CVXPY not available)
        """
        n = self.n_assets
        
        # Initial weights
        if current_weights is not None:
            x0 = current_weights.values
        else:
            x0 = np.ones(n) / n
        
        def objective(w):
            # Expected return
            port_return = np.dot(w, self.expected_returns)
            
            # Risk
            port_risk = np.sqrt(np.dot(w, np.dot(self.cov_matrix, w)))
            
            # Liquidity penalty
            liquidity_penalty = 0
            for i, symbol in enumerate(self.expected_returns.index):
                liquidity_score = self.calculate_liquidity_score(symbol)
                # Penalize low liquidity positions
                liquidity_penalty += w[i] * (1 - liquidity_score) * 0.01
            
            # Sharpe ratio with liquidity adjustment
            sharpe = (port_return - 0.02 - liquidity_penalty) / port_risk
            
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Bounds
        bounds = []
        for i, symbol in enumerate(self.expected_returns.index):
            liquidity_score = self.calculate_liquidity_score(symbol)
            
            # Tighter bounds for illiquid assets
            if liquidity_score < 0.3:
                max_weight = 0.05
            elif liquidity_score < 0.6:
                max_weight = 0.15
            else:
                max_weight = 0.30
            
            bounds.append((0, max_weight))
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        # Extract results
        optimal_weights = pd.Series(result.x, index=self.expected_returns.index)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(self.cov_matrix, optimal_weights)))
        liquidity_metrics = self.calculate_portfolio_liquidity(optimal_weights, portfolio_value)
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': (portfolio_return - 0.02) / portfolio_risk,
            'liquidity_metrics': liquidity_metrics,
            'optimization_status': 'optimal' if result.success else 'failed'
        }
    
    def calculate_liquidity_risk_metrics(self, weights: pd.Series, 
                                       portfolio_value: float,
                                       stress_multiplier: float = 3.0) -> Dict[str, float]:
        """
        Calculate liquidity risk metrics including stressed scenarios
        """
        position_values = weights * portfolio_value
        
        # Normal market liquidation cost
        normal_impact = 0
        for symbol, position in position_values.items():
            if abs(position) > 0:
                impact = self.impact_model.almgren_chriss_impact(
                    symbol, abs(position), time_horizon=1.0
                )
                normal_impact += impact['total_cost']
        
        # Stressed market liquidation cost
        stressed_impact = 0
        for symbol, position in position_values.items():
            if abs(position) > 0:
                # In stressed conditions, assume wider spreads and lower volumes
                stressed_params = self.impact_model.impact_params[symbol]
                original_spread = stressed_params.spread
                original_volume = stressed_params.daily_volume
                
                # Temporarily modify parameters for stress scenario
                stressed_params.spread *= stress_multiplier
                stressed_params.daily_volume /= stress_multiplier
                
                impact = self.impact_model.almgren_chriss_impact(
                    symbol, abs(position), time_horizon=1.0
                )
                stressed_impact += impact['total_cost']
                
                # Restore original parameters
                stressed_params.spread = original_spread
                stressed_params.daily_volume = original_volume
        
        # Liquidity Value at Risk (LVaR)
        # Assumes log-normal distribution of liquidation costs
        confidence_level = 0.95
        lvar = normal_impact * np.exp(norm.ppf(confidence_level) * 0.5)  # 50% volatility assumption
        
        return {
            'normal_liquidation_cost': normal_impact,
            'stressed_liquidation_cost': stressed_impact,
            'liquidity_var_95': lvar,
            'liquidity_stress_ratio': stressed_impact / normal_impact if normal_impact > 0 else np.inf,
            'liquidation_cost_pct': (normal_impact / portfolio_value) * 100
        }

def create_liquidity_metrics_from_data(prices_df: pd.DataFrame,
                                     volumes_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
    """
    Helper function to create liquidity metrics from price/volume data
    """
    liquidity_metrics = {}
    
    for symbol in prices_df.columns:
        returns = prices_df[symbol].pct_change().dropna()
        
        # Basic metrics
        current_price = prices_df[symbol].iloc[-1]
        volatility = returns.std()
        
        # Volume (use provided or estimate)
        if volumes_df is not None and symbol in volumes_df.columns:
            adv = volumes_df[symbol].mean()
        else:
            # Rough estimate based on market cap assumptions
            adv = 1_000_000 * (100 / current_price)
        
        # Spread estimation (if not available)
        # Rule of thumb: spread ≈ 2 * daily_vol / sqrt(252)
        estimated_spread = 2 * volatility / np.sqrt(252)
        
        # Depth estimation (proportional to ADV)
        estimated_depth = adv / 1000  # Rough estimate
        
        liquidity_metrics[symbol] = {
            'adv': adv,
            'spread': estimated_spread,
            'volatility': volatility,
            'depth': estimated_depth,
            'price': current_price
        }
    
    return liquidity_metrics