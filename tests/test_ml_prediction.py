"""
Test script for ML price prediction module.

This demonstrates how to use the ML predictor for enhanced portfolio optimization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher import DataFetcher
from src.ml.price_predictor import MLPricePredictor, RNNPricePredictor, MLEnhancedOptimizer
from src.optimization.mean_variance import MeanVarianceOptimizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("ML-Enhanced Portfolio Optimization Demo")
    print("=" * 50)
    
    # Step 1: Fetch data
    print("\n1. Fetching historical data...")
    fetcher = DataFetcher()
    
    # Use a few stocks for quick demonstration
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = fetcher.fetch_price_data(tickers, '2018-01-01', '2024-01-01')
    
    print(f"   ✓ Fetched {len(prices)} days of data for {len(tickers)} stocks")
    
    # Step 2: Train ML model for one stock
    print("\n2. Training ML model for AAPL...")
    
    # Prepare data
    aapl_data = prices[['AAPL']].copy()
    aapl_data.columns = ['Close']
    
    # Initialize predictor
    ml_predictor = MLPricePredictor(model_type='ensemble')
    
    # Prepare and train
    ml_data = ml_predictor.prepare_data(aapl_data, train_size=0.8)
    result = ml_predictor.train_ensemble(ml_data)
    
    print(f"   ✓ Model trained successfully")
    print(f"   ✓ Ensemble MSE: {result.model_metrics['ensemble_mse']:.6f}")
    
    # Step 3: Display feature importance
    print("\n3. Top 10 Most Important Features:")
    top_features = result.feature_importance.head(10)
    for idx, (feature, importance) in enumerate(top_features.iterrows(), 1):
        print(f"   {idx}. {feature}: {importance['importance']:.4f}")
    
    # Step 4: Compare predictions
    print("\n4. Model Predictions Comparison:")
    for model in result.predictions.columns:
        mse = result.model_metrics.get(f'{model}_mse', np.nan)
        print(f"   - {model}: MSE = {mse:.6f}")
    
    # Step 5: Train models for all assets
    print("\n5. Training models for all assets...")
    all_predictions = {}
    
    for ticker in tickers:
        print(f"   Training {ticker}...")
        asset_data = prices[[ticker]].copy()
        asset_data.columns = ['Close']
        
        predictor = MLPricePredictor()
        data = predictor.prepare_data(asset_data, train_size=0.8)
        pred_result = predictor.train_ensemble(data)
        
        all_predictions[ticker] = pred_result.predictions['ensemble'].mean() * 252
        print(f"   ✓ {ticker}: Expected annual return = {all_predictions[ticker]:.2%}")
    
    # Step 6: ML-Enhanced optimization
    print("\n6. Running ML-Enhanced Portfolio Optimization...")
    
    # Traditional optimization first
    trad_optimizer = MeanVarianceOptimizer()
    trad_result = trad_optimizer.optimize(prices, objective='max_sharpe')
    
    print("\n   Traditional Optimization:")
    print(f"   - Expected Return: {trad_result.expected_return:.2%}")
    print(f"   - Volatility: {trad_result.volatility:.2%}")
    print(f"   - Sharpe Ratio: {trad_result.sharpe_ratio:.3f}")
    
    # ML-enhanced optimization
    returns = prices.pct_change().dropna()
    historical_mean = returns.mean() * 252
    historical_cov = returns.cov() * 252
    
    # Blend ML and historical estimates
    ml_expected_returns = pd.Series(all_predictions)
    blended_returns = 0.6 * ml_expected_returns + 0.4 * historical_mean
    
    # Create enhanced optimizer
    ml_enhanced_optimizer = MLEnhancedOptimizer(ml_predictor, confidence_weight=0.6)
    ml_result = ml_enhanced_optimizer.optimize_with_ml(
        prices, 
        historical_window=252,
        optimization_method='max_sharpe'
    )
    
    print("\n   ML-Enhanced Optimization:")
    print(f"   - Expected Return: {ml_result['expected_return']:.2%}")
    print(f"   - Volatility: {ml_result['volatility']:.2%}")
    print(f"   - Sharpe Ratio: {ml_result['sharpe_ratio']:.3f}")
    
    # Step 7: Compare portfolios
    print("\n7. Portfolio Weights Comparison:")
    print("   " + "-" * 40)
    print(f"   {'Asset':<10} {'Traditional':<15} {'ML-Enhanced':<15}")
    print("   " + "-" * 40)
    
    for i, ticker in enumerate(tickers):
        trad_weight = trad_result.weights[i]
        ml_weight = ml_result['weights'][i]
        print(f"   {ticker:<10} {trad_weight:<15.2%} {ml_weight:<15.2%}")
    
    # Step 8: Quick RNN demonstration
    print("\n8. Training LSTM model (this may take a minute)...")
    
    # Prepare sequences for LSTM
    from sklearn.preprocessing import MinMaxScaler
    
    # Create features
    from src.ml.price_predictor import FeatureEngineer
    fe = FeatureEngineer()
    features = fe.create_technical_features(aapl_data).dropna()
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    sequence_length = 60
    if len(scaled_features) > sequence_length + 10:
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(scaled_features) - 5):
            X_seq.append(scaled_features[i-sequence_length:i])
            # 5-day return
            future_price = aapl_data['Close'].iloc[i + 5]
            current_price = aapl_data['Close'].iloc[i]
            y_seq.append((future_price / current_price - 1))
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Train/test split
        split = int(0.8 * len(X_seq))
        X_train, y_train = X_seq[:split], y_seq[:split]
        X_val = X_train[-50:]  # Last 50 samples for validation
        y_val = y_train[-50:]
        
        # Train simple LSTM
        rnn = RNNPricePredictor(sequence_length=sequence_length)
        history = rnn.train(
            X_train[:-50], y_train[:-50],  # Training
            X_val, y_val,  # Validation
            architecture='gru',  # GRU is faster
            epochs=20,  # Reduced for demo
            batch_size=32
        )
        
        print("   ✓ LSTM trained successfully")
        print(f"   ✓ Final validation loss: {history['val_loss'][-1]:.6f}")
    
    print("\n" + "=" * 50)
    print("ML-Enhanced Portfolio Optimization Complete!")
    print("\nKey Insights:")
    print("- ML predictions can improve expected return estimates")
    print("- Feature engineering is crucial for good predictions")
    print("- Ensemble models reduce prediction variance")
    print("- LSTM/RNN can capture temporal patterns in prices")
    print("\nNext steps: Run full backtesting to validate performance")

if __name__ == "__main__":
    main()