"""
Black-Litterman Model Implementation

The Black-Litterman model combines market equilibrium with investor views
to generate more stable, intuitive expected returns for portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class BlackLittermanModel:
    """
    Black-Litterman Model for generating expected returns.
    
    This model starts with equilibrium returns implied by market capitalization
    weights and adjusts them based on investor views with specified confidence levels.
    """
    
    def __init__(
        self,
        cov_matrix: pd.DataFrame,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Black-Litterman model.
        
        Args:
            cov_matrix: Covariance matrix of asset returns (annualized)
            risk_aversion: Market risk aversion parameter (typically 2-4)
                         Higher = more risk averse market
            tau: Uncertainty in the prior (typically 0.01-0.05)
                 Represents uncertainty in equilibrium returns
            risk_free_rate: Annual risk-free rate
        """
        self.cov_matrix = cov_matrix
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.assets = list(cov_matrix.index)
        self.n_assets = len(self.assets)
        
        # These will be set when calculating equilibrium
        self.market_weights = None
        self.equilibrium_returns = None
        
    def calculate_equilibrium_returns(
        self, 
        market_weights: Union[pd.Series, Dict[str, float]]
    ) -> pd.Series:
        """
        Calculate equilibrium returns using reverse optimization.
        
        The key insight: If the market is in equilibrium and everyone
        uses mean-variance optimization, what returns would lead to
        the current market weights?
        
        Formula: Π = λ * Σ * w_mkt
        where:
            Π = equilibrium returns
            λ = risk aversion
            Σ = covariance matrix
            w_mkt = market capitalization weights
        
        Args:
            market_weights: Market cap weights (must sum to 1)
            
        Returns:
            Series of equilibrium expected returns
        """
        # Convert to Series if dict
        if isinstance(market_weights, dict):
            market_weights = pd.Series(market_weights)
            
        # Ensure weights are aligned with covariance matrix
        market_weights = market_weights.reindex(self.assets).fillna(0)
        
        # Normalize to sum to 1
        market_weights = market_weights / market_weights.sum()
        self.market_weights = market_weights
        
        # Calculate equilibrium returns
        # This is the return that would make current weights optimal
        self.equilibrium_returns = self.risk_aversion * self.cov_matrix @ market_weights
        
        return self.equilibrium_returns
    
    def create_views(
        self,
        views: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create view matrices from view specifications.
        
        Views can be:
        1. Absolute: "AAPL will return 15%"
        2. Relative: "AAPL will outperform MSFT by 3%"
        
        Args:
            views: List of view dictionaries, each containing:
                - 'type': 'absolute' or 'relative'
                - 'assets': asset(s) involved
                - 'return': expected return (absolute) or outperformance (relative)
                - 'confidence': confidence level (0-1, where 1 is 100% confident)
                
        Example:
            views = [
                {
                    'type': 'absolute',
                    'assets': ['AAPL'],
                    'return': 0.15,  # 15% expected return
                    'confidence': 0.8  # 80% confident
                },
                {
                    'type': 'relative', 
                    'assets': ['AAPL', 'MSFT'],
                    'weights': [1, -1],  # AAPL minus MSFT
                    'return': 0.03,  # 3% outperformance
                    'confidence': 0.6  # 60% confident
                }
            ]
            
        Returns:
            P: Pick matrix (which assets are in each view)
            Q: View returns vector
            Omega: Uncertainty matrix (diagonal)
        """
        n_views = len(views)
        
        # Initialize matrices
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)
        
        for i, view in enumerate(views):
            view_type = view['type']
            assets = view['assets']
            view_return = view['return']
            confidence = view['confidence']
            
            # Get asset indices
            asset_indices = [self.assets.index(asset) for asset in assets]
            
            if view_type == 'absolute':
                # Absolute view: asset will have specific return
                P[i, asset_indices[0]] = 1
                Q[i] = view_return
                
            elif view_type == 'relative':
                # Relative view: asset1 will outperform asset2
                weights = view.get('weights', [1, -1])  # Default: asset1 - asset2
                for j, (idx, weight) in enumerate(zip(asset_indices, weights)):
                    P[i, idx] = weight
                Q[i] = view_return
                
            else:
                raise ValueError(f"Unknown view type: {view_type}")
            
            # Calculate uncertainty from confidence
            # Lower confidence = higher uncertainty
            # We use the variance of the view portfolio
            view_portfolio_var = P[i] @ self.cov_matrix @ P[i].T
            
            # Scale by tau and inverse confidence
            # confidence=1 → very low uncertainty
            # confidence=0.5 → moderate uncertainty  
            omega_diag[i] = view_portfolio_var * self.tau / confidence
        
        # Create diagonal uncertainty matrix
        Omega = np.diag(omega_diag)
        
        return P, Q, Omega
    
    def calculate_posterior_returns(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray
    ) -> pd.Series:
        """
        Calculate posterior expected returns using Black-Litterman formula.
        
        This is the mathematical heart of Black-Litterman:
        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        
        Args:
            P: Pick matrix
            Q: View returns
            Omega: View uncertainty matrix
            
        Returns:
            Posterior (blended) expected returns
        """
        if self.equilibrium_returns is None:
            raise ValueError("Must calculate equilibrium returns first")
        
        # Prior (equilibrium) precision and mean
        tau_sigma_inv = np.linalg.inv(self.tau * self.cov_matrix)
        prior_precision = tau_sigma_inv
        prior_mean = tau_sigma_inv @ self.equilibrium_returns
        
        # View precision and mean
        omega_inv = np.linalg.inv(Omega)
        view_precision = P.T @ omega_inv @ P
        view_mean = P.T @ omega_inv @ Q
        
        # Posterior precision and mean
        posterior_precision = prior_precision + view_precision
        posterior_mean = prior_mean + view_mean
        
        # Posterior expected returns
        posterior_returns = np.linalg.inv(posterior_precision) @ posterior_mean
        
        # Also calculate posterior covariance (useful for confidence intervals)
        self.posterior_cov = np.linalg.inv(posterior_precision)
        
        return pd.Series(posterior_returns, index=self.assets)
    
    def get_confidence_intervals(
        self, 
        posterior_returns: pd.Series,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Calculate confidence intervals for posterior returns.
        
        This shows the uncertainty in our expected returns after
        incorporating views.
        
        Args:
            posterior_returns: Posterior expected returns
            confidence_level: Confidence level for intervals
            
        Returns:
            DataFrame with lower and upper bounds
        """
        # Standard errors from diagonal of posterior covariance
        posterior_std = np.sqrt(np.diag(self.posterior_cov))
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate bounds
        lower_bound = posterior_returns - z_score * posterior_std
        upper_bound = posterior_returns + z_score * posterior_std
        
        return pd.DataFrame({
            'expected_return': posterior_returns,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std_error': posterior_std
        })
    
    def optimize_portfolio(
        self,
        posterior_returns: pd.Series,
        target_return: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> pd.Series:
        """
        Optimize portfolio using Black-Litterman posterior returns.
        
        This is just standard mean-variance optimization, but with
        better (more stable) expected returns.
        
        Args:
            posterior_returns: Expected returns from Black-Litterman
            target_return: Target portfolio return (None for max Sharpe)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            
        Returns:
            Optimal portfolio weights
        """
        from scipy.optimize import minimize
        
        n = len(posterior_returns)
        
        # Objective: minimize negative Sharpe ratio
        def negative_sharpe(weights):
            port_return = weights @ posterior_returns
            port_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            return -(port_return - self.risk_free_rate) / port_vol
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ posterior_returns - target_return
            })
        
        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(n)]
        
        # Initial guess (equal weights)
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            negative_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        return pd.Series(result.x, index=posterior_returns.index)