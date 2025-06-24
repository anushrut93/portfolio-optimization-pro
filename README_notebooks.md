# Portfolio Optimization Notebooks

This directory contains a comprehensive series of Jupyter notebooks that demonstrate the complete quantitative portfolio management pipeline, from data exploration through production-ready strategies.

## 📚 Notebook Overview

### 1. **01_data_exploration.ipynb**
**Purpose**: Initial data analysis and exploratory data analysis (EDA)

**Key Topics**:
- Loading historical price data for AAPL, GOOGL, MSFT, AMZN, JPM (2015-2024)
- Calculating returns and key statistics
- Correlation analysis and visualization
- Sector analysis and risk characteristics
- Data quality checks and preprocessing

**Key Outputs**:
- Processed price and return datasets
- Correlation matrices
- Summary statistics (annual returns, volatility, Sharpe ratios)

**Dependencies**: None (starting point)

---

### 2. **02_portfolio_theory.ipynb**
**Purpose**: Implementation of Modern Portfolio Theory concepts

**Key Topics**:
- Efficient frontier construction
- Markowitz mean-variance optimization
- Maximum Sharpe ratio portfolio
- Minimum volatility portfolio
- Risk parity implementation
- Black-Litterman model basics

**Key Results**:
- Traditional Max Sharpe: 1.039 Sharpe ratio
- Min Volatility: 0.932 Sharpe ratio
- Efficient frontier visualization
- Optimal weight allocations

**Dependencies**: 01_data_exploration.ipynb (for processed data)

---

### 3. **02a_ml_price_prediction.ipynb**
**Purpose**: Machine learning and deep learning for return prediction

**Key Topics**:
- Feature engineering (50+ technical indicators)
- Random Forest and XGBoost implementation
- **LSTM/RNN neural networks** for sequence modeling
- Ensemble model creation
- Walk-forward validation framework
- Feature importance analysis

**Key Results**:
- LSTM MSE: 0.001498 (best individual model)
- Ensemble R²: 0.68
- Directional accuracy: 58.3%
- 45% Sharpe ratio improvement over traditional methods

**Dependencies**: 01_data_exploration.ipynb

**Special Features**:
- GPU-accelerated LSTM training
- Attention mechanisms for temporal patterns
- Multi-horizon forecasting (1, 5, 20 days)

---

### 4. **03_Portfolio_Strategy_Implementation_&_Trading_Cost_Analysis.ipynb**
**Purpose**: Comprehensive backtesting with realistic constraints

**Key Topics**:
- Strategy implementation and comparison
- Transaction cost analysis
- Rebalancing frequency optimization
- Monte Carlo simulation
- Black-Litterman with ML views
- Performance attribution

**Key Results**:
- ML-Enhanced Sharpe: 1.256 (21% improvement)
- Optimal rebalancing: Monthly
- Transaction cost impact: ~10-20% on returns
- Win rate: 84.2% in Monte Carlo simulations

**Dependencies**: 02_portfolio_theory.ipynb, 02a_ml_price_prediction.ipynb

---

### 5. **04_risk_analysis.ipynb**
**Purpose**: Comprehensive risk assessment and stress testing

**Key Topics**:
- Value at Risk (VaR) calculations
- Conditional VaR (CVaR)
- Stress testing scenarios
- Drawdown analysis
- Correlation analysis during crises
- Portfolio stability metrics

**Key Results**:
- 95% VaR: -2.31% daily
- Maximum drawdown: -16.1% to -20.7% by strategy
- Crisis correlations spike to 0.9+
- Tail risk analysis with fat-tail distributions

**Dependencies**: All previous notebooks

---

### 6. **05_garch_volatility_forecasting.ipynb**
**Purpose**: Advanced volatility modeling and forecasting

**Key Topics**:
- GARCH(1,1) implementation
- Model comparison (GARCH-N, GARCH-t, GJR-GARCH, EGARCH)
- Dynamic correlation analysis (DCC-GARCH)
- Volatility forecasting
- Risk parity with GARCH weights
- Volatility trading strategies

**Key Results**:
- GJR-GARCH selected (lowest AIC: 5643.95)
- Volatility strategy Sharpe: 0.884
- 20.6% improvement over buy-and-hold
- Superior risk forecasting during crises

**Dependencies**: 01_data_exploration.ipynb

**Note**: AMZN excluded from GARCH due to convergence issues

---

### 7. **06_market_microstructure_analysis.ipynb**
**Purpose**: Liquidity analysis and execution optimization

**Key Topics**:
- Order book simulation and analysis
- Market impact modeling (Almgren-Chriss)
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Liquidity-aware portfolio optimization
- Implementation shortfall analysis
- Real-time monitoring system design

**Key Results**:
- 16.8% reduction in implementation costs
- $750K annual savings on $1B AUM
- Liquidity-aware Sharpe: 1.037
- Production monitoring dashboard

**Dependencies**: Previous optimization notebooks

**Unique Features**:
- Order flow toxicity detection
- Bid-ask spread decomposition
- Optimal execution trajectories

---

### 8. **07_long_short_strategy.ipynb**
**Purpose**: Hedge fund-style market-neutral strategies

**Key Topics**:
- Alpha factor development (momentum, reversal, quality)
- Long/short portfolio construction
- Market neutrality constraints
- Parameter optimization
- Regime-specific performance analysis
- Super Alpha strategy development

**Key Results**:
- Best Sharpe (COVID period): 1.38
- Optimal configuration: 4 long + 4 short positions
- Monthly rebalancing with 3 bps costs
- 74% win rate across different regimes

**Dependencies**: All previous notebooks

**Production Ready**: Includes full implementation guide

---

## 🚀 Getting Started

### Recommended Order
1. Start with `01_data_exploration.ipynb` to understand the data
2. Progress through `02_portfolio_theory.ipynb` for foundational concepts
3. Explore ML enhancements in `02a_ml_price_prediction.ipynb`
4. See everything come together in `03_Portfolio_Strategy_Implementation_&_Trading_Cost_Analysis.ipynb`
5. Deep dive into specific areas (risk, volatility, microstructure, alpha) based on interest

### Running the Notebooks

```bash
# Ensure you're in the project root directory
cd portfolio-optimization

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/

# Or for specific notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Data Requirements
- All notebooks use the same base dataset (2015-2024)
- Data is fetched automatically using `yfinance`
- Processed data is saved in CSV format for consistency

### Computational Requirements
- Most notebooks run on standard hardware
- `02a_ml_price_prediction.ipynb` benefits from GPU for LSTM training
- `06_market_microstructure_analysis.ipynb` is memory intensive due to order book simulation

---

## 📊 Key Results Summary

| Notebook | Primary Metric | Result |
|----------|---------------|---------|
| 01 - Data Exploration | Asset correlations | 0.3-0.7 normal, 0.9+ crisis |
| 02 - Portfolio Theory | Traditional Sharpe | 1.039 |
| 02a - ML Prediction | Directional accuracy | 58.3% |
| 03 - Implementation | ML-Enhanced Sharpe | 1.256 (+21%) |
| 04 - Risk Analysis | 95% VaR | -2.31% daily |
| 05 - GARCH | Volatility forecast | 40% improvement |
| 06 - Microstructure | Cost reduction | 16.8% ($750K/year) |
| 07 - Long/Short | Alpha Sharpe | 1.38 (best regime) |

---

## 💡 Tips for Exploration

### For Researchers
- Focus on notebooks 02a (ML), 05 (GARCH), and 07 (Alpha strategies)
- Experiment with different feature sets in the ML notebook
- Try alternative volatility models in notebook 05

### For Practitioners
- Start with notebook 03 for implementation details
- Notebook 06 is crucial for real-world trading
- Pay attention to transaction cost analysis

### For Risk Managers
- Notebooks 04 and 05 provide comprehensive risk frameworks
- Crisis period analysis shows strategy robustness
- Stress testing results guide position sizing

---

## 🔧 Customization

Each notebook is designed to be modular. You can:
- Change the asset universe (modify ticker lists)
- Adjust time periods (change start/end dates)
- Modify parameters (risk aversion, constraints, costs)
- Add new strategies or models

---

## 📝 Notes

- **Reproducibility**: Random seeds are set where applicable
- **Performance**: Results may vary slightly due to data updates
- **Dependencies**: Some notebooks save results used by others
- **Best Practice**: Run notebooks in order for first time

For questions or issues, refer to the main project README or raise an issue on GitHub.