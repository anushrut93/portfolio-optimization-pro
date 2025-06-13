# Portfolio Optimization Notebooks

This directory contains Jupyter notebooks that demonstrate and explain the portfolio optimization process in detail.

## Notebook Structure

### 1. **01_data_exploration.ipynb**
- Load and explore historical price data
- Visualize price trends and correlations
- Calculate basic statistics (returns, volatility)
- Identify any data quality issues
- Test different data sources

### 2. **02_portfolio_theory.ipynb**
- Explain Modern Portfolio Theory concepts
- Demonstrate efficient frontier calculation
- Show the mathematics behind optimization
- Interactive widgets to adjust parameters
- Visualize risk-return tradeoffs

### 3. **03_optimization_methods.ipynb**
- Compare different optimization strategies:
  - Mean-Variance (Markowitz)
  - Minimum Volatility
  - Maximum Sharpe Ratio
  - Risk Parity
  - Black-Litterman
- Show optimization constraints in action
- Sensitivity analysis

### 4. **04_backtesting_analysis.ipynb**
- Walk through the backtesting process
- Analyze transaction costs impact
- Compare rebalancing frequencies
- Out-of-sample performance analysis
- Rolling window optimization

### 5. **05_risk_analysis.ipynb**
- Deep dive into risk metrics
- Value at Risk (VaR) and CVaR
- Stress testing and scenario analysis
- Drawdown analysis
- Factor exposure analysis

### 6. **06_ml_forecasting.ipynb** (Advanced)
- Return prediction using ML models
- Feature engineering for financial data
- Model validation and backtesting
- Ensemble methods
- Integration with optimization

### 7. **07_live_dashboard.ipynb**
- Real-time portfolio monitoring
- Interactive parameter adjustment
- Performance attribution
- What-if scenarios
- Export results

## Usage

1. Install required packages:
```bash
pip install jupyter notebook pandas numpy matplotlib seaborn plotly yfinance scipy scikit-learn
```

2. Start Jupyter:
```bash
jupyter notebook
```

