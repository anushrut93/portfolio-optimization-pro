import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

"""
Data fetcher module for portfolio optimization.

This module handles all data acquisition from various sources,
with robust error handling and data validation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    A robust data fetcher for financial market data.
    
    This class provides methods to fetch historical price data,
    with support for multiple tickers, error handling, and data validation.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the DataFetcher.
        
        Args:
            cache_dir: Optional directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        self._yahoo_finance_delay = 0.1  # Delay between requests to be respectful
        
    def fetch_price_data(
        self,
        tickers: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical price data for given tickers.
        
        This is the main method users will interact with. It handles both
        single tickers and lists of tickers, converts date formats, and
        returns clean data ready for analysis.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            interval: Data frequency ('1d', '1wk', '1mo')
            
        Returns:
            DataFrame with price data, tickers as columns, dates as index
            
        Raises:
            ValueError: If no valid data is retrieved
        """
        # Convert single ticker to list for uniform processing
        if isinstance(tickers, str):
            tickers = [tickers]
            
        # Convert dates to datetime objects if they're strings
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) if end_date else datetime.now()
        
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}")
        
        # Fetch data with error handling
        price_data = self._fetch_with_retry(tickers, start_date, end_date, interval)
        
        # Validate and clean the data
        cleaned_data = self._validate_and_clean(price_data)
        
        if cleaned_data.empty:
            raise ValueError("No valid price data retrieved")
            
        return cleaned_data
    
    def _fetch_with_retry(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch data with retry logic for robustness.
        
        This method implements exponential backoff - if a request fails,
        it waits progressively longer before retrying. This helps handle
        temporary network issues or API rate limits.
        """
        all_data = pd.DataFrame()
        failed_tickers = []
        
        # Use ThreadPoolExecutor for parallel downloads (faster for many tickers)
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Create a future for each ticker
            future_to_ticker = {
                executor.submit(
                    self._fetch_single_ticker,
                    ticker,
                    start_date,
                    end_date,
                    interval,
                    max_retries
                ): ticker
                for ticker in tickers
            }
            
            # Process completed futures
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        all_data[ticker] = data['Adj Close']
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Failed to fetch {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to fetch data for: {failed_tickers}")
            
        return all_data
    
    def _fetch_single_ticker(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        max_retries: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker with retry logic.
        
        This method demonstrates good API citizenship by implementing
        exponential backoff and respecting rate limits.
        """
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid overwhelming the API
                time.sleep(self._yahoo_finance_delay)
                
                # Download the data
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
                
                if not data.empty:
                    logger.info(f"Successfully fetched {ticker}")
                    return data
                    
            except Exception as e:
                wait_time = (2 ** attempt) * self._yahoo_finance_delay
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
        return None
    
    def _validate_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the fetched data.
        
        This method ensures data quality by:
        1. Removing entirely empty columns (failed tickers)
        2. Forward filling missing values (weekends/holidays)
        3. Removing any remaining NaN values
        4. Checking for data anomalies
        """
        if data.empty:
            return data
            
        # Remove columns with all NaN values
        data = data.dropna(axis=1, how='all')
        
        # Forward fill missing values (common for weekends/holidays)
        data = data.fillna(method='ffill')
        
        # Drop any remaining rows with NaN values
        initial_rows = len(data)
        data = data.dropna()
        
        if len(data) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(data)} rows with missing data")
            
        # Check for suspicious data (e.g., prices <= 0)
        for col in data.columns:
            if (data[col] <= 0).any():
                logger.warning(f"Found non-positive prices in {col}")
                data = data[data[col] > 0]
                
        return data
    
    def fetch_market_data(
        self,
        market_index: str = '^GSPC',
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None
    ) -> pd.Series:
        """
        Fetch market index data (default: S&P 500).
        
        This is used for CAPM calculations and market comparisons.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=5*365)  # 5 years default
            
        market_data = self.fetch_price_data(market_index, start_date, end_date)
        return market_data.squeeze()  # Convert single-column DataFrame to Series


# Example usage and testing
if __name__ == "__main__":
    # This section only runs when the file is executed directly
    # It's a good practice to include examples in your modules
    
    fetcher = DataFetcher()
    
    # Example 1: Fetch single stock
    print("Fetching Apple stock data...")
    apple_data = fetcher.fetch_price_data('AAPL', '2022-01-01', '2023-01-01')
    print(f"Retrieved {len(apple_data)} days of data")
    print(apple_data.head())
    
    # Example 2: Fetch multiple stocks
    print("\nFetching portfolio stocks...")
    portfolio_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    portfolio_data = fetcher.fetch_price_data(portfolio_stocks, '2022-01-01')
    print(f"Retrieved data for {len(portfolio_data.columns)} stocks")
    print(portfolio_data.tail())