#!/usr/bin/env python3
"""
Example script showing how to run the portfolio optimization pipeline
"""

import subprocess
import os
import json

def run_basic_example():
    """Run a basic example with default settings"""
    print("Running basic portfolio optimization...")
    
    # Run with command line arguments
    cmd = [
        "python", "main.py",
        "--tickers", "AAPL", "GOOGL", "MSFT", "AMZN", "JPM",
        "--start-date", "2020-01-01",
        "--end-date", "2023-12-31",
        "--initial-capital", "100000"
    ]
    
    subprocess.run(cmd)
    
def run_with_config():
    """Run with a configuration file"""
    print("Running portfolio optimization with config file...")
    
    # Create a custom config
    config = {
        "tickers": ["SPY", "AGG", "GLD", "VNQ", "EFA", "EEM"],  # Diversified ETF portfolio
        "start_date": "2018-01-01",
        "end_date": "2023-12-31",
        "train_end_date": "2021-12-31",
        "initial_capital": 500000,
        "risk_free_rate": 0.03,
        "rebalance_frequency": "monthly",
        "commission": 0.0005,
        "constraints": {
            "min_weight": 0.05,  # Minimum 5% in any position
            "max_weight": 0.35   # Maximum 35% in any position
        }
    }
    
    # Save config
    with open("custom_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Run with config
    cmd = ["python", "main.py", "--config", "custom_config.json"]
    subprocess.run(cmd)
    
def run_sector_rotation():
    """Run a sector rotation strategy"""
    print("Running sector rotation portfolio optimization...")
    
    # Sector ETFs
    config = {
        "tickers": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"],
        "start_date": "2019-01-01",
        "end_date": "2023-12-31",
        "train_end_date": "2022-06-30",
        "initial_capital": 250000,
        "risk_free_rate": 0.025,
        "target_return": 0.12,
        "rebalance_frequency": "monthly",
        "commission": 0.0,  # No commission for this example
        "constraints": {
            "min_weight": 0.0,   # Allow zero weights
            "max_weight": 0.20,  # Maximum 20% in any sector
            "min_positions": 5   # At least 5 sectors
        }
    }
    
    with open("sector_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    cmd = ["python", "main.py", "--config", "sector_config.json"]
    subprocess.run(cmd)

def analyze_results():
    """Read and display the results"""
    print("\n" + "="*50)
    print("ANALYZING RESULTS")
    print("="*50 + "\n")
    
    # Read the summary report
    if os.path.exists("output/portfolio_summary.txt"):
        with open("output/portfolio_summary.txt", "r") as f:
            print(f.read())
    
    # Read the JSON report for more details
    if os.path.exists("output/portfolio_report.json"):
        with open("output/portfolio_report.json", "r") as f:
            report = json.load(f)
            
        print("\nDETAILED METRICS:")
        print("-"*50)
        
        for strategy, metrics in report["risk_metrics"].items():
            print(f"\n{strategy}:")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  VaR (95%): {metrics['var_95']:.2%}")
            print(f"  CVaR (95%): {metrics['cvar_95']:.2%}")
    
    print("\nVisualization files generated in output/visualizations/")
    if os.path.exists("output/visualizations"):
        files = os.listdir("output/visualizations")
        for f in files:
            print(f"  - {f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "basic":
            run_basic_example()
        elif example == "config":
            run_with_config()
        elif example == "sector":
            run_sector_rotation()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python run_example.py [basic|config|sector]")
    else:
        # Run all examples
        print("Running all examples...\n")
        run_basic_example()
        print("\n" + "="*70 + "\n")
        run_with_config()
        print("\n" + "="*70 + "\n")
        run_sector_rotation()
    
    # Always analyze results at the end
    analyze_results()