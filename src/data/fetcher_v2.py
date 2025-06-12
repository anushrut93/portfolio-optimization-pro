"""
Data fetcher module for portfolio optimization - Version 2.
This version is compatible with the latest yfinance library.
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
    
    This class provides methods to fetch historical price data,
    with support for multiple tickers and error handling.
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
        else:
            ticker_list = tickers
            
        # Convert dates to strings (yfinance prefers string dates)
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        logger.info(f"Fetching data for {ticker_list} from {start_date} to {end_date or 'today'}")
        
        try:
            # Method 1: Try downloading all at once (most efficient)
            if len(ticker_list) == 1:
                # For single ticker, download directly
                data = yf.download(
                    ticker_list[0],
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # If successful, extract Adjusted Close and create DataFrame
                if not data.empty and 'Adj Close' in data.columns:
                    result = pd.DataFrame({ticker_list[0]: data['Adj Close']})
                elif not data.empty and 'Close' in data.columns:
                    # Fallback to Close if Adj Close not available
                    result = pd.DataFrame({ticker_list[0]: data['Close']})
                else:
                    raise ValueError(f"No price data found for {ticker_list[0]}")
                    
            else:
                # For multiple tickers
                data = yf.download(
                    ticker_list,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by='ticker'
                )
                
                # Extract Adjusted Close prices for each ticker
                result = pd.DataFrame()
                
                # When multiple tickers are downloaded, yfinance returns MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    for ticker in ticker_list:
                        try:
                            if (ticker, 'Adj Close') in data.columns:
                                result[ticker] = data[(ticker, 'Adj Close')]
                            elif (ticker, 'Close') in data.columns:
                                result[ticker] = data[(ticker, 'Close')]
                        except:
                            logger.warning(f"Could not extract data for {ticker}")
                else:
                    # Single ticker result
                    if 'Adj Close' in data.columns:
                        result = pd.DataFrame({ticker_list[0]: data['Adj Close']})
                    elif 'Close' in data.columns:
                        result = pd.DataFrame({ticker_list[0]: data['Close']})
            
            # Clean the data
            result = result.dropna()
            
            if result.empty:
                raise ValueError("No valid data retrieved after cleaning")
                
            logger.info(f"Successfully retrieved data: {result.shape[0]} rows, {result.shape[1]} columns")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            
            # Method 2: If batch download fails, try one by one
            logger.info("Attempting to fetch tickers individually...")
            result = pd.DataFrame()
            
            for ticker in ticker_list:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(
                        start=start_date,
                        end=end_date,
                        auto_adjust=True
                    )
                    
                    if not hist.empty:
                        result[ticker] = hist['Close']
                        logger.info(f"Successfully fetched {ticker}")
                    else:
                        logger.warning(f"No data for {ticker}")
                        
                except Exception as ticker_error:
                    logger.error(f"Failed to fetch {ticker}: {str(ticker_error)}")
            
            if result.empty:
                raise ValueError("Could not retrieve data for any ticker")
                
            return result