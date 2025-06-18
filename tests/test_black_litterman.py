"""
Unit tests for Black-Litterman model
"""

import unittest
import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from src.optimization.black_litterman import BlackLittermanModel


class TestBlackLitterman(unittest.TestCase):
    """Test cases for Black-Litterman implementation"""
    
    def setUp(self):
        """Set up test data"""
        # Create simple test data
        self.assets = ['A', 'B', 'C']
        self.cov_matrix = pd.DataFrame(
            [[0.04, 0.01, 0.02],
             [0.01, 0.09, 0.01],
             [0.02, 0.01, 0.16]],
            index=self.assets,
            columns=self.assets
        )
        
        self.market_weights = pd.Series([0.5, 0.3, 0.2], index=self.assets)
        
        self.bl_model = BlackLittermanModel(
            cov_matrix=self.cov_matrix,
            risk_aversion=2.5,
            tau=0.05
        )
    
    def test_equilibrium_returns(self):
        """Test equilibrium return calculation"""
        eq_returns = self.bl_model.calculate_equilibrium_returns(self.market_weights)
        
        # Check basic properties
        self.assertEqual(len(eq_returns), 3)
        self.assertTrue(all(eq_returns > 0))  # Should be positive for positive weights
        
        # Verify calculation
        expected = 2.5 * self.cov_matrix @ self.market_weights
        np.testing.assert_array_almost_equal(eq_returns.values, expected.values)
    
    def test_absolute_view(self):
        """Test absolute view creation"""
        self.bl_model.calculate_equilibrium_returns(self.market_weights)
        
        views = [{
            'type': 'absolute',
            'assets': ['A'],
            'return': 0.10,
            'confidence': 0.8
        }]
        
        P, Q, Omega = self.bl_model.create_views(views)
        
        # Check dimensions
        self.assertEqual(P.shape, (1, 3))
        self.assertEqual(len(Q), 1)
        self.assertEqual(Omega.shape, (1, 1))
        
        # Check view matrix
        np.testing.assert_array_equal(P[0], [1, 0, 0])
        self.assertEqual(Q[0], 0.10)
    
    def test_relative_view(self):
        """Test relative view creation"""
        self.bl_model.calculate_equilibrium_returns(self.market_weights)
        
        views = [{
            'type': 'relative',
            'assets': ['A', 'B'],
            'weights': [1, -1],
            'return': 0.05,
            'confidence': 0.6
        }]
        
        P, Q, Omega = self.bl_model.create_views(views)
        
        # Check view matrix
        np.testing.assert_array_equal(P[0], [1, -1, 0])
        self.assertEqual(Q[0], 0.05)
    
    def test_posterior_returns(self):
        """Test posterior return calculation"""
        eq_returns = self.bl_model.calculate_equilibrium_returns(self.market_weights)
        
        # Strong view on asset A
        views = [{
            'type': 'absolute',
            'assets': ['A'],
            'return': 0.20,  # Much higher than equilibrium
            'confidence': 0.9
        }]
        
        P, Q, Omega = self.bl_model.create_views(views)
        posterior = self.bl_model.calculate_posterior_returns(P, Q, Omega)
        
        # Asset A should have higher return than equilibrium
        self.assertGreater(posterior['A'], eq_returns['A'])
        
        # Other assets should also adjust (correlation effect)
        self.assertNotEqual(posterior['B'], eq_returns['B'])
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization with BL returns"""
        eq_returns = self.bl_model.calculate_equilibrium_returns(self.market_weights)
        
        # Optimize
        weights = self.bl_model.optimize_portfolio(eq_returns)
        
        # Check constraints
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertTrue(all(weights >= 0))
        self.assertTrue(all(weights <= 1))


if __name__ == '__main__':
    unittest.main()