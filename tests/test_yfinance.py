"""
Tests for yfinance integration.

Updated to handle MultiIndex columns properly.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetcher import DataFetcher


class TestYfinanceIntegration:
    """Test yfinance data fetching functionality"""
    
    @pytest.fixture
    def fetcher(self):
        """Create DataFetcher instance"""
        return DataFetcher()
    
    def test_single_ticker_fetch(self, fetcher):
        """Test fetching data for a single ticker"""
        # Mock yfinance download to avoid actual API calls
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Adj Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with patch('yfinance.download', return_value=mock_data):
            result = fetcher.fetch_price_data('AAPL', '2023-01-01', '2023-01-03')
            
            assert isinstance(result, pd.DataFrame)
            assert 'AAPL' in result.columns
            assert len(result) == 3
    
    def test_multiple_tickers_fetch(self, fetcher):
        """Test fetching data for multiple tickers"""
        # Create mock data with MultiIndex columns (like real yfinance)
        dates = pd.date_range('2023-01-01', periods=3)
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        # Create MultiIndex for columns
        columns = pd.MultiIndex.from_product([
            ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
            tickers
        ])
        
        # Create mock data
        data = np.random.randn(3, 18)  # 3 days × 6 fields × 3 tickers
        mock_data = pd.DataFrame(data, index=dates, columns=columns)
        
        # Set some realistic values for Adj Close
        for ticker in tickers:
            mock_data[('Adj Close', ticker)] = [100 + i for i in range(3)]
        
        with patch('yfinance.download', return_value=mock_data):
            result = fetcher.fetch_price_data(tickers, '2023-01-01', '2023-01-03')
            
            assert isinstance(result, pd.DataFrame)
            assert all(ticker in result.columns for ticker in tickers)
            assert len(result) == 3
    
    def test_empty_data_handling(self, fetcher):
        """Test handling of empty data from yfinance"""
        empty_data = pd.DataFrame()
        
        with patch('yfinance.download', return_value=empty_data):
            with pytest.raises(ValueError, match="Could not retrieve data for any ticker"):
                fetcher.fetch_price_data('INVALID', '2023-01-01', '2023-01-03')
    
    def test_multiindex_extraction(self, fetcher):
        """Test proper extraction of Adj Close from MultiIndex"""
        dates = pd.date_range('2023-01-01', periods=3)
        tickers = ['AAPL', 'MSFT']
        
        # Create proper MultiIndex structure
        columns = pd.MultiIndex.from_tuples([
            ('Adj Close', 'AAPL'),
            ('Adj Close', 'MSFT'),
            ('Close', 'AAPL'),
            ('Close', 'MSFT'),
            ('Volume', 'AAPL'),
            ('Volume', 'MSFT')
        ])
        
        data = {
            ('Adj Close', 'AAPL'): [100, 101, 102],
            ('Adj Close', 'MSFT'): [200, 201, 202],
            ('Close', 'AAPL'): [99, 100, 101],
            ('Close', 'MSFT'): [199, 200, 201],
            ('Volume', 'AAPL'): [1000000, 1100000, 1200000],
            ('Volume', 'MSFT'): [2000000, 2100000, 2200000]
        }
        
        mock_data = pd.DataFrame(data, index=dates)
        
        with patch('yfinance.download', return_value=mock_data):
            result = fetcher.fetch_price_data(tickers, '2023-01-01', '2023-01-03')
            
            # Check that we extracted the right data
            assert result['AAPL'].iloc[0] == 100
            assert result['MSFT'].iloc[0] == 200
            assert list(result.columns) == ['AAPL', 'MSFT']
    
    def test_fallback_to_close_prices(self, fetcher):
        """Test fallback to Close prices when Adj Close not available"""
        dates = pd.date_range('2023-01-01', periods=3)
        tickers = ['AAPL', 'MSFT']
        
        # Create MultiIndex without Adj Close
        columns = pd.MultiIndex.from_tuples([
            ('Close', 'AAPL'),
            ('Close', 'MSFT'),
            ('Volume', 'AAPL'),
            ('Volume', 'MSFT')
        ])
        
        data = {
            ('Close', 'AAPL'): [100, 101, 102],
            ('Close', 'MSFT'): [200, 201, 202],
            ('Volume', 'AAPL'): [1000000, 1100000, 1200000],
            ('Volume', 'MSFT'): [2000000, 2100000, 2200000]
        }
        
        mock_data = pd.DataFrame(data, index=dates)
        
        with patch('yfinance.download', return_value=mock_data):
            with patch('logging.Logger.info') as mock_logger:
                result = fetcher.fetch_price_data(tickers, '2023-01-01', '2023-01-03')
                
                # Should use Close prices
                assert result['AAPL'].iloc[0] == 100
                assert result['MSFT'].iloc[0] == 200
    
    def test_auto_adjust_parameter(self, fetcher):
        """Test that auto_adjust is set to False"""
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = pd.DataFrame({'Adj Close': [100]}, 
                                                    index=pd.date_range('2023-01-01', periods=1))
            
            fetcher.fetch_price_data('AAPL', '2023-01-01', '2023-01-03')
            
            # Check that auto_adjust=False was passed
            _, kwargs = mock_download.call_args
            assert kwargs.get('auto_adjust') == False
    
    def test_date_parsing(self, fetcher):
        """Test date parsing functionality"""
        from datetime import datetime
        
        mock_data = pd.DataFrame({
            'Adj Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with patch('yfinance.download', return_value=mock_data):
            # Test with datetime objects
            result = fetcher.fetch_price_data(
                'AAPL',
                datetime(2023, 1, 1),
                datetime(2023, 1, 3)
            )
            assert len(result) == 3
    
    def test_ticker_error_handling(self, fetcher):
        """Test individual ticker error handling in fallback"""
        dates = pd.date_range('2023-01-01', periods=3)
        
        # Create a mock that succeeds for individual tickers
        def mock_ticker_history(*args, **kwargs):
            mock_hist = pd.DataFrame({
                'Close': [100, 101, 102]
            }, index=dates)
            return mock_hist
        
        # First call to download fails, then individual calls succeed
        with patch('yfinance.download', side_effect=Exception("Download failed")):
            with patch('yfinance.Ticker') as mock_ticker_class:
                mock_ticker = MagicMock()
                mock_ticker.history.return_value = mock_ticker_history()
                mock_ticker_class.return_value = mock_ticker
                
                result = fetcher.fetch_price_data(['AAPL', 'MSFT'], '2023-01-01', '2023-01-03')
                
                assert 'AAPL' in result.columns
                assert 'MSFT' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])