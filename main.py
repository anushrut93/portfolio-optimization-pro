#!/usr/bin/env python3
"""
Main Portfolio Optimization Pipeline
====================================
This script demonstrates the complete workflow:
1. Fetch historical data
2. Run optimization
3. Perform backtesting
4. Calculate risk metrics
5. Generate visualizations
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import your modules
try:
    from data.fetcher import DataFetcher
except ImportError as e:
    logger.error(f"Could not import DataFetcher: {e}")
    
try:
    from optimization.mean_variance import MeanVarianceOptimizer
except ImportError as e:
    logger.error(f"Could not import MeanVarianceOptimizer: {e}")
    
try:
    from backtesting.engine import BacktestEngine, BacktestConfig, RebalanceFrequency
    
    # Create a wrapper to match the expected interface
    class BacktestingEngine:
        def __init__(self, initial_capital=100000, commission=0.001):
            self.config = BacktestConfig(
                initial_capital=initial_capital,
                transaction_cost_pct=commission,
                rebalance_frequency=RebalanceFrequency.QUARTERLY
            )
            self.engine = BacktestEngine(self.config)
            
        def run_backtest(self, prices, weights, rebalance_frequency='quarterly'):
            # Convert rebalance frequency string to enum
            freq_map = {
                'daily': RebalanceFrequency.DAILY,
                'weekly': RebalanceFrequency.WEEKLY,
                'monthly': RebalanceFrequency.MONTHLY,
                'quarterly': RebalanceFrequency.QUARTERLY,
                'yearly': RebalanceFrequency.YEARLY
            }
            self.config.rebalance_frequency = freq_map.get(rebalance_frequency, RebalanceFrequency.QUARTERLY)
            
            # Create a strategy function that returns the fixed weights
            def fixed_weight_strategy(historical_prices):
                # Ensure weights are aligned with price columns
                return pd.Series(weights, index=prices.columns).values
            
            # Run the backtest
            results = self.engine.run_backtest(prices, fixed_weight_strategy)
            
            # Convert results to expected format
            return {
                'portfolio_value': results.portfolio_values,
                'returns': results.returns,
                'trades': results.trades
            }
            
except ImportError as e:
    logger.error(f"Could not import backtesting module: {e}")
    
try:
    from visualization.plots import PortfolioVisualizer
except ImportError as e:
    logger.error(f"Could not import PortfolioVisualizer: {e}")


class PortfolioPipeline:
    """Main pipeline for portfolio optimization workflow"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.optimizer = None
        self.backtest_results = None
        self.risk_metrics = None
        
    def run(self):
        """Execute the complete pipeline"""
        logger.info("Starting Portfolio Optimization Pipeline")
        
        # Step 1: Fetch Data
        self.fetch_data()
        
        # Step 2: Optimize Portfolio
        self.optimize_portfolio()
        
        # Step 3: Backtest Strategy
        self.run_backtest()
        
        # Step 4: Calculate Risk Metrics
        self.calculate_risk_metrics()
        
        # Step 5: Generate Visualizations
        self.create_visualizations()
        
        # Step 6: Generate Report
        self.generate_report()
        
        logger.info("Pipeline completed successfully")
        
    def fetch_data(self):
    	"""Fetch historical price data"""
    	logger.info(f"Fetching data for tickers: {self.config['tickers']}")
    
    	fetcher = DataFetcher()
    
    	# fetch_price_data can handle multiple tickers at once
    	try:
            self.prices = fetcher.fetch_price_data(
            tickers=self.config['tickers'],
            start_date=self.config['start_date'],
            end_date=self.config['end_date']
            )
        
            # Handle different column formats
            if isinstance(self.prices.columns, pd.MultiIndex):
            	# Extract just the ticker names from MultiIndex
            	new_columns = []
            	for col in self.prices.columns:
                	if isinstance(col, tuple) and len(col) == 2:
                    		ticker, field = col
                    		if field in ['Adj Close', 'Close']:
                        		new_columns.append(ticker)
                	else:
                    		new_columns.append(col)
            	self.prices.columns = new_columns
            elif 'Adj Close' in self.prices.columns:
            	self.prices = self.prices.rename(columns={'Adj Close': self.config['tickers'][0]})
            elif 'Close' in self.prices.columns:
            	self.prices = self.prices.rename(columns={'Close': self.config['tickers'][0]})
            
            self.returns = self.prices.pct_change().dropna()
            logger.info(f"Data shape: {self.prices.shape}")
            logger.info(f"Successfully fetched data for {len(self.prices.columns)} tickers")
    	except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise ValueError(f"No data fetched successfully: {e}")
            
    def optimize_portfolio(self):
        """Run portfolio optimization"""
        logger.info("Running portfolio optimization")
        
        # Initialize optimizer
        self.optimizer = MeanVarianceOptimizer(
            risk_free_rate=self.config.get('risk_free_rate', 0.02)
        )
        
        # Get optimal weights for different objectives
        self.optimization_results = {}
        
        # Maximum Sharpe Ratio
        result = self.optimizer.optimize(
            prices=self.prices,
            objective='max_sharpe',
            constraints=self.config.get('constraints', {})
        )
        self.optimization_results['max_sharpe'] = {
            'weights': pd.Series(result.weights, index=result.asset_names),
            'expected_return': result.expected_return,
            'expected_risk': result.volatility,
            'sharpe_ratio': result.sharpe_ratio
        }
        
        # Minimum Volatility
        result = self.optimizer.optimize(
            prices=self.prices,
            objective='min_volatility',
            constraints=self.config.get('constraints', {})
        )
        self.optimization_results['min_volatility'] = {
            'weights': pd.Series(result.weights, index=result.asset_names),
            'expected_return': result.expected_return,
            'expected_risk': result.volatility,
            'sharpe_ratio': result.sharpe_ratio
        }
            
        logger.info("Optimization completed")
        
    def run_backtest(self):
        """Backtest the optimized portfolio"""
        logger.info("Running backtest")
        
        # Split data into train and test
        split_date = pd.to_datetime(self.config.get('train_end_date', 
                                                   self.prices.index[int(len(self.prices) * 0.7)]))
        
        train_prices = self.prices[self.prices.index <= split_date]
        test_prices = self.prices[self.prices.index > split_date]
        
        # Initialize backtesting engine
        engine = BacktestingEngine(
            initial_capital=self.config.get('initial_capital', 100000),
            commission=self.config.get('commission', 0.001)
        )
        
        self.backtest_results = {}
        
        # Backtest each optimization strategy
        for strategy_name, weights in self.optimization_results.items():
            logger.info(f"Backtesting {strategy_name} strategy")
            
            results = engine.run_backtest(
                prices=test_prices,
                weights=weights['weights'],
                rebalance_frequency=self.config.get('rebalance_frequency', 'quarterly')
            )
            
            self.backtest_results[strategy_name] = results
            
        # Also backtest equal weight benchmark
        equal_weights = pd.Series(
            1.0 / len(self.config['tickers']),
            index=self.config['tickers']
        )
        self.backtest_results['equal_weight'] = engine.run_backtest(
            prices=test_prices,
            weights=equal_weights,
            rebalance_frequency=self.config.get('rebalance_frequency', 'quarterly')
        )
        
        logger.info("Backtesting completed")
        
    def calculate_risk_metrics(self):
        """Calculate risk metrics for all strategies"""
        logger.info("Calculating risk metrics")
        
        from risk.metrics import RiskAnalyzer
        risk_analyzer = RiskAnalyzer(risk_free_rate=self.config.get('risk_free_rate', 0.02))
        
        self.risk_metrics = {}
        
        for strategy_name, backtest_result in self.backtest_results.items():
            portfolio_returns = backtest_result['returns']
            portfolio_values = backtest_result['portfolio_value']
            
            # Use RiskAnalyzer to get comprehensive metrics
            risk_result = risk_analyzer.analyze_risk(portfolio_values)
            
            # Convert to dictionary format expected by the rest of the code
            metrics = {
                'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1),
                'annual_return': risk_result.volatility * 0,  # Will be recalculated
                'annual_volatility': risk_result.volatility,
                'sharpe_ratio': risk_result.sharpe_ratio,
                'sortino_ratio': risk_result.sortino_ratio,
                'max_drawdown': risk_result.max_drawdown,
                'var_95': risk_result.var_95,
                'cvar_95': risk_result.cvar_95
            }
            
            # Calculate annualized return properly
            years = len(portfolio_values) / 252
            metrics['annual_return'] = (1 + metrics['total_return']) ** (1/years) - 1
            
            self.risk_metrics[strategy_name] = metrics
            
        logger.info("Risk metrics calculated")
        
    def create_visualizations(self):
        """Generate portfolio visualizations"""
        logger.info("Creating visualizations")
        
        visualizer = PortfolioVisualizer(style='professional')
        
        # Create output directory if it doesn't exist
        os.makedirs('output/visualizations', exist_ok=True)
        
        # Performance comparison chart
        first_strategy = list(self.backtest_results.keys())[0]
        portfolio_series = self.backtest_results[first_strategy]['portfolio_value']
        
        # Get benchmark if we have equal weight strategy
        benchmark = None
        if 'equal_weight' in self.backtest_results:
            benchmark = self.backtest_results['equal_weight']['portfolio_value']
        
        visualizer.plot_portfolio_performance(
            portfolio_values=portfolio_series,
            benchmark_values=benchmark,
            save_path="output/visualizations/performance_comparison.png"
        )
        
        # Risk-return scatter plot
        assets_data = {}
        for strategy_name, metrics in self.risk_metrics.items():
            assets_data[strategy_name] = {
                'return': metrics['annual_return'],
                'volatility': metrics['annual_volatility']
            }
        
        visualizer.plot_risk_return_scatter(
            assets_data=assets_data,
            save_path="output/visualizations/risk_return_scatter.png"
        )
        
        # Portfolio weights for each strategy
        for strategy_name, results in self.optimization_results.items():
            weights_dict = results['weights'].to_dict()
            visualizer.plot_asset_allocation(
                weights=weights_dict,
                title=f"Portfolio Weights - {strategy_name}",
                save_path=f"output/visualizations/weights_{strategy_name}.png"
            )
            
        logger.info("Visualizations created")
        
    def generate_report(self):
        """Generate a comprehensive report"""
        logger.info("Generating report")
        
        report = {
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'optimization_results': {
                strategy: {
                    'weights': weights['weights'].to_dict(),
                    'expected_return': weights.get('expected_return'),
                    'expected_risk': weights.get('expected_risk')
                }
                for strategy, weights in self.optimization_results.items()
            },
            'risk_metrics': self.risk_metrics,
            'backtest_summary': {
                strategy: {
                    'final_value': float(results['portfolio_value'].iloc[-1]),
                    'total_return': float((results['portfolio_value'].iloc[-1] / 
                                         results['portfolio_value'].iloc[0] - 1) * 100),
                    'number_of_trades': len(results.get('trades', []))
                }
                for strategy, results in self.backtest_results.items()
            }
        }
        
        # Save report as JSON
        with open('output/portfolio_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        # Generate summary text report
        with open('output/portfolio_summary.txt', 'w') as f:
            f.write("PORTFOLIO OPTIMIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run Date: {report['run_date']}\n")
            f.write(f"Tickers: {', '.join(self.config['tickers'])}\n")
            f.write(f"Period: {self.config['start_date']} to {self.config['end_date']}\n\n")
            
            f.write("STRATEGY PERFORMANCE SUMMARY\n")
            f.write("-" * 50 + "\n")
            
            for strategy in self.risk_metrics:
                f.write(f"\n{strategy.upper()}:\n")
                f.write(f"  Annual Return: {self.risk_metrics[strategy]['annual_return']:.2%}\n")
                f.write(f"  Annual Volatility: {self.risk_metrics[strategy]['annual_volatility']:.2%}\n")
                f.write(f"  Sharpe Ratio: {self.risk_metrics[strategy]['sharpe_ratio']:.3f}\n")
                f.write(f"  Max Drawdown: {self.risk_metrics[strategy]['max_drawdown']:.2%}\n")
                
        logger.info("Report generated")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Portfolio Optimization Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JPM'],
                       help='List of ticker symbols')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial capital')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Use command line arguments
        config = {
            'tickers': args.tickers,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'initial_capital': args.initial_capital,
            'train_end_date': '2022-12-31',  # Use 2020-2022 for training, 2023 for testing
            'risk_free_rate': 0.02,
            'rebalance_frequency': 'quarterly',
            'commission': 0.001,
            'constraints': {
                'min_weight': 0.0,
                'max_weight': 0.4  # Maximum 40% in any single asset
            }
        }
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/visualizations', exist_ok=True)
    
    # Run pipeline
    pipeline = PortfolioPipeline(config)
    pipeline.run()
    
    logger.info("All done! Check the output directory for results.")


if __name__ == '__main__':
    main()