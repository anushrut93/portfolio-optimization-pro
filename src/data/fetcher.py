"""
Fixed Data fetcher module that properly handles yfinance MultiIndex columns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Union, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    A robust data fetcher for financial market data.
    Fixed to handle MultiIndex columns from yfinance correctly.
    """
    
    def __init__(self):
        """Initialize the DataFetcher."""
        logger.info("DataFetcher initialized")
        
    def fetch_price_data(
        self,
        tickers: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data for given tickers.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with adjusted close prices
        """
        # Convert single ticker to list for uniform processing
        if isinstance(tickers, str):
            ticker_list = [tickers]
            single_ticker = True
        else:
            ticker_list = tickers
            single_ticker = False
            
        # Convert dates to strings (yfinance prefers string dates)
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if end_date is not None and isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        # If end_date is None, yfinance will default to today
            
        logger.info(f"Fetching data for {ticker_list} from {start_date} to {end_date or 'today'}")
        
        try:
            # Download data with auto_adjust=False to get Adj Close column
            data = yf.download(
                ticker_list,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )
            
            if data.empty:
                raise ValueError("No data retrieved from yfinance")
            
            # Handle the MultiIndex columns issue
            if isinstance(data.columns, pd.MultiIndex):
                # For multiple tickers, extract Adj Close using the working method
                result = pd.DataFrame()
                
                # Get all columns for inspection
                for col in data.columns:
                    if isinstance(col, tuple) and len(col) == 2:
                        field, ticker = col[0], col[1]
                        if field == 'Adj Close':
                            result[ticker] = data[(field, ticker)]
                
                if result.empty:
                    # Fallback: try to get Close prices if Adj Close not available
                    for col in data.columns:
                        if isinstance(col, tuple) and len(col) == 2:
                            field, ticker = col[0], col[1]
                            if field == 'Close':
                                result[ticker] = data[(field, ticker)]
                                
            else:
                # Single ticker case
                if 'Adj Close' in data.columns:
                    result = pd.DataFrame({ticker_list[0]: data['Adj Close']})
                elif 'Close' in data.columns:
                    result = pd.DataFrame({ticker_list[0]: data['Close']})
                else:
                    raise ValueError("No price data found in downloaded data")
            
            # Clean the data
            result = result.dropna()
            
            if result.empty:
                raise ValueError("No valid data after cleaning")
                
            logger.info(f"Successfully retrieved data: {result.shape[0]} rows, {result.shape[1]} columns")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            
            # Try alternative approach: fetch one by one
            logger.info("Attempting to fetch tickers individually...")
            result = pd.DataFrame()
            
            for ticker in ticker_list:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(
                        start=start_date,
                        end=end_date,
                        auto_adjust=True  # This gives us adjusted prices directly
                    )
                    
                    if not hist.empty:
                        result[ticker] = hist['Close']  # With auto_adjust=True, Close is adjusted
                        logger.info(f"Successfully fetched {ticker}")
                    else:
                        logger.warning(f"No data for {ticker}")
                        
                except Exception as ticker_error:
                    logger.error(f"Failed to fetch {ticker}: {str(ticker_error)}")
            
            if result.empty:
                raise ValueError("Could not retrieve data for any ticker")
                
            return result.dropna()


# Test the fixed fetcher
if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Test with multiple tickers
    print("Testing with multiple tickers...")
    try:
        data = fetcher.fetch_price_data(
            ['AAPL', 'GOOGL', 'MSFT'],
            '2023-01-01',
            '2023-01-31'
        )
        print(f"Success! Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")