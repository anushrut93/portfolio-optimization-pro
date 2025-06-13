"""
Simple test script to verify our data fetcher works.
Run this from the project root directory.
"""

import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can import our fetcher
from src.data.fetcher_v2 import DataFetcher

def main():
    print("Testing DataFetcher...")
    print("-" * 50)
    
    # Create a fetcher instance
    fetcher = DataFetcher()
    
    # Test 1: Single stock, recent data
    print("\nTest 1: Fetching recent Apple data...")
    try:
        data = fetcher.fetch_price_data('AAPL', '2024-01-01', '2024-02-01')
        print(f"Success! Retrieved {len(data)} days of data")
        print("\nFirst few rows:")
        print(data.head())
        print("\nData info:")
        print(f"- Date range: {data.index[0]} to {data.index[-1]}")
        print(f"- Average price: ${data['AAPL'].mean():.2f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Multiple stocks
    print("\n" + "-" * 50)
    print("\nTest 2: Fetching multiple stocks...")
    try:
        stocks = ['MSFT', 'GOOGL']
        data = fetcher.fetch_price_data(stocks, '2024-01-01', '2024-01-31')
        print(f"Success! Retrieved data for {len(data.columns)} stocks")
        print("\nLast few rows:")
        print(data.tail())
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()