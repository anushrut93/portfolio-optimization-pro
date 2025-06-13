"""
Test script for the data fetcher module.

This demonstrates how to properly import and use the DataFetcher
from outside the module itself.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# When we import from src, Python knows where to look

from src.data.fetcher import DataFetcher
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def test_single_stock():
    """Test fetching data for a single stock."""
    print("Testing single stock fetch...")
    fetcher = DataFetcher()
    
    try:
        # Fetch Apple stock data
        data = fetcher.fetch_price_data('AAPL', '2022-01-01', '2022-02-01')
        print(f"✓ Successfully fetched {len(data)} days of AAPL data")
        print(f"✓ Date range: {data.index[0]} to {data.index[-1]}")
        print(f"✓ Columns: {data.columns.tolist()}")
        print("\nFirst 5 rows:")
        print(data.head())
    except Exception as e:
        print(f"✗ Error: {e}")

def test_multiple_stocks():
    """Test fetching data for multiple stocks."""
    print("\n\nTesting multiple stock fetch...")
    fetcher = DataFetcher()
    
    try:
        stocks = ['AAPL', 'MSFT', 'GOOGL']
        data = fetcher.fetch_price_data(stocks, '2022-01-01', '2022-01-31')
        print(f"✓ Successfully fetched data for {len(data.columns)} stocks")
        print(f"✓ Stocks retrieved: {data.columns.tolist()}")
        print("\nLast 5 rows:")
        print(data.tail())
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_single_stock()
    test_multiple_stocks()