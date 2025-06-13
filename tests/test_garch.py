"""
Tests for GARCH volatility modeling module.

This test suite covers:
- Model fitting and parameter estimation
- Volatility forecasting
- Portfolio weight calculations
- Dynamic strategy backtesting
- Error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.volatility.garch import (
    GARCHModeler, 
    GARCHResults, 
    DynamicGARCHStrategy
)
from tests.fixtures import generate_test_returns


class TestGARCHResults:
    """Test the GARCHResults dataclass"""
    
    def test_garch_results_creation(self):
        """Test creating GARCHResults instance"""
        results = GARCHResults(
            model_type='GARCH',
            parameters={'omega': 0.0001, 'alpha': 0.1, 'beta': 0.8},
            conditional_volatility=pd.Series([0.01, 0.015, 0.02]),
            volatility_forecast=np.array([0.018, 0.017, 0.016]),
            forecast_horizon=3,
            aic=1000.5,
            bic=1010.5,
            log_likelihood=-495.5
        )
        
        assert results.model_type == 'GARCH'
        assert results.parameters['alpha'] == 0.1
        assert len(results.volatility_forecast) == 3
        
    def test_annualized_forecast(self):
        """Test volatility annualization"""
        daily_vol = np.array([0.01, 0.015, 0.02])
        results = GARCHResults(
            model_type='GARCH',
            parameters={},
            conditional_volatility=pd.Series([]),
            volatility_forecast=daily_vol,
            forecast_horizon=3,
            aic=0,
            bic=0,
            log_likelihood=0
        )
        
        annual_vol = results.annualized_forecast()
        expected = daily_vol * np.sqrt(252)
        np.testing.assert_array_almost_equal(annual_vol, expected)


class TestGARCHModeler:
    """Test the main GARCHModeler class"""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data"""
        return generate_test_returns(n_assets=3, n_periods=500)
    
    @pytest.fixture
    def modeler(self, sample_returns):
        """Create GARCHModeler instance"""
        return GARCHModeler(sample_returns)
    
    def test_initialization(self, sample_returns):
        """Test GARCHModeler initialization"""
        modeler = GARCHModeler(sample_returns)
        assert modeler.returns_data.equals(sample_returns)
        assert len(modeler.fitted_models) == 0
        assert len(modeler.results) == 0
    
    def test_fit_garch_single_series(self, modeler, sample_returns):
        """Test fitting GARCH to single series"""
        asset = 'ASSET1'
        result = modeler.fit_garch(sample_returns[asset])
        
        assert isinstance(result, GARCHResults)
        assert result.model_type == 'GARCH'
        assert 'omega' in result.parameters
        assert 'alpha' in result.parameters
        assert 'beta' in result.parameters
        assert len(result.conditional_volatility) == len(sample_returns)
        
    def test_fit_garch_different_models(self, modeler, sample_returns):
        """Test fitting different GARCH specifications"""
        asset = 'ASSET1'
        
        # Test GARCH
        garch_result = modeler.fit_garch(
            sample_returns[asset], 
            model_type='GARCH'
        )
        assert garch_result.model_type == 'GARCH'
        
        # Test GJR-GARCH
        gjr_result = modeler.fit_garch(
            sample_returns[asset], 
            model_type='GJR-GARCH',
            o=1
        )
        assert gjr_result.model_type == 'GJR-GARCH'
        assert 'gamma' in gjr_result.parameters
        
        # Test EGARCH
        egarch_result = modeler.fit_garch(
            sample_returns[asset], 
            model_type='EGARCH'
        )
        assert egarch_result.model_type == 'EGARCH'
    
    def test_forecast_volatility(self, modeler, sample_returns):
        """Test volatility forecasting"""
        asset = 'ASSET1'
        
        # Fit model first
        modeler.fit_garch(sample_returns[asset])
        
        # Generate forecast
        horizon = 10
        forecast = modeler.forecast_volatility(asset, horizon=horizon)
        
        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == horizon
        assert np.all(forecast > 0)  # Volatility should be positive
        
    def test_forecast_without_fitting(self, modeler):
        """Test error when forecasting without fitting"""
        with pytest.raises(ValueError, match="No fitted model"):
            modeler.forecast_volatility('ASSET1')
    
    def test_fit_all_assets(self, modeler):
        """Test fitting models to all assets"""
        results = modeler.fit_all_assets(model_type='GARCH')
        
        assert len(results) == 3  # 3 assets
        for asset, result in results.items():
            assert isinstance(result, GARCHResults)
            assert asset in modeler.fitted_models
    
    def test_calculate_risk_parity_weights(self, modeler):
        """Test risk parity weight calculation"""
        # Fit models first
        modeler.fit_all_assets()
        
        # Calculate weights
        weights = modeler.calculate_risk_parity_weights(forecast_horizon=5)
        
        assert isinstance(weights, pd.Series)
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-6  # Sum to 1
        assert np.all(weights > 0)  # All positive
    
    def test_calculate_min_variance_weights(self, modeler):
        """Test minimum variance weight calculation"""
        modeler.fit_all_assets()
        
        # Test with GARCH volatilities
        weights_garch = modeler.calculate_min_variance_weights(use_garch_vol=True)
        assert np.abs(weights_garch.sum() - 1.0) < 1e-6
        
        # Test without GARCH volatilities
        weights_simple = modeler.calculate_min_variance_weights(use_garch_vol=False)
        assert np.abs(weights_simple.sum() - 1.0) < 1e-6
        
        # Weights should be different
        assert not np.allclose(weights_garch, weights_simple)
    
    def test_detect_volatility_regime(self, modeler, sample_returns):
        """Test volatility regime detection"""
        asset = 'ASSET1'
        modeler.fit_garch(sample_returns[asset])
        
        regime = modeler.detect_volatility_regime(asset)
        assert regime in ['low', 'normal', 'high']
        
        # Test with custom thresholds
        regime_custom = modeler.detect_volatility_regime(
            asset, 
            low_threshold=0.1, 
            high_threshold=0.9
        )
        assert regime_custom in ['low', 'normal', 'high']
    
    def test_calculate_var_cvar(self, modeler):
        """Test VaR and CVaR calculation"""
        modeler.fit_all_assets()
        
        # Equal weights
        weights = pd.Series(1/3, index=modeler.returns_data.columns)
        
        var, cvar = modeler.calculate_var_cvar(
            weights, 
            confidence_level=0.95, 
            horizon=1
        )
        
        assert isinstance(var, float)
        assert isinstance(cvar, float)
        assert cvar < var  # CVaR should be more negative than VaR
    
    def test_compare_models(self, modeler):
        """Test model comparison functionality"""
        comparison = modeler.compare_models()
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'Asset' in comparison.columns
        assert 'Model' in comparison.columns
        assert 'AIC' in comparison.columns
        assert 'BIC' in comparison.columns
        
        # Should have results for 3 assets × 3 models = 9 rows (if all succeed)
        assert len(comparison) <= 9


class TestDynamicGARCHStrategy:
    """Test the DynamicGARCHStrategy class"""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance"""
        returns = generate_test_returns(n_assets=3, n_periods=500)
        modeler = GARCHModeler(returns)
        modeler.fit_all_assets()
        return DynamicGARCHStrategy(modeler)
    
    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.garch_modeler is not None
        assert isinstance(strategy.base_weights, pd.Series)
        assert np.abs(strategy.base_weights.sum() - 1.0) < 1e-6
    
    def test_calculate_dynamic_weights_risk_parity(self, strategy):
        """Test risk parity weight calculation"""
        weights = strategy.calculate_dynamic_weights(method='risk_parity')
        
        assert isinstance(weights, pd.Series)
        assert np.abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights > 0)
    
    def test_calculate_dynamic_weights_min_variance(self, strategy):
        """Test minimum variance weight calculation"""
        # Ensure GARCH models are fitted first
        strategy.garch_modeler.fit_all_assets()
        
        weights = strategy.calculate_dynamic_weights(method='min_variance')
        
        assert weights is not None
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-6
    
    def test_calculate_dynamic_weights_volatility_scaling(self, strategy):
        """Test volatility scaling weight calculation"""
        # Without target
        weights = strategy.calculate_dynamic_weights(method='volatility_scaling')
        assert isinstance(weights, pd.Series)
        
        # With volatility target
        weights_targeted = strategy.calculate_dynamic_weights(
            method='volatility_scaling',
            volatility_target=0.15
        )
        assert isinstance(weights_targeted, pd.Series)
    
    def test_unknown_method(self, strategy):
        """Test error for unknown method"""
        with pytest.raises(ValueError, match="Unknown method"):
            strategy.calculate_dynamic_weights(method='unknown_method')
    
    def test_backtest_strategy(self, strategy):
        """Test strategy backtesting"""
        results = strategy.backtest_strategy(
            rebalance_frequency='monthly',
            method='risk_parity'
        )
        
        assert 'returns' in results
        assert 'cumulative_returns' in results
        assert 'annual_return' in results
        assert 'annual_volatility' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        
        # Check metrics are reasonable
        assert -1 < results['annual_return'] < 2  # Reasonable range
        assert 0 < results['annual_volatility'] < 1
        assert -1 < results['max_drawdown'] < 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_asset_portfolio(self):
        """Test with single asset"""
        returns = generate_test_returns(n_assets=1, n_periods=300)
        modeler = GARCHModeler(returns)
        
        results = modeler.fit_all_assets()
        assert len(results) == 1
        
        # Risk parity should give 100% weight
        weights = modeler.calculate_risk_parity_weights()
        assert weights.iloc[0] == 1.0
    
    def test_short_time_series(self):
        """Test with short time series"""
        returns = generate_test_returns(n_assets=2, n_periods=50)
        modeler = GARCHModeler(returns)
        
        # Should still work but might have warnings
        result = modeler.fit_garch(returns.iloc[:, 0])
        assert isinstance(result, GARCHResults)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        returns = generate_test_returns(n_assets=2, n_periods=200)
        
        # Introduce some NaN values
        returns.iloc[50:55, 0] = np.nan
        
        modeler = GARCHModeler(returns.dropna())
        result = modeler.fit_garch(returns.iloc[:, 0].dropna())
        
        assert isinstance(result, GARCHResults)
        assert not result.conditional_volatility.isna().any()


class TestIntegration:
    """Integration tests with other modules"""
    
    
    def test_arch_model_integration(self):
        """Test real integration with arch library"""
        # Use real arch library - no mocking
        returns = generate_test_returns(n_assets=1, n_periods=200)
        modeler = GARCHModeler(returns)
        
        # Name the series  
        returns.iloc[:, 0].name = 'TEST_ASSET'
        
        # This will use the real arch_model
        result = modeler.fit_garch(returns.iloc[:, 0], dist='normal')  # Use normal for faster fitting
        
        # Verify we get valid results
        assert isinstance(result, GARCHResults)
        assert 'alpha' in result.parameters
        assert 'beta' in result.parameters
        assert result.parameters['alpha'] > 0  # Should be positive
        assert result.parameters['beta'] > 0   # Should be positive
        assert result.aic is not None
        assert result.bic is not None
        assert len(result.conditional_volatility) == len(returns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])