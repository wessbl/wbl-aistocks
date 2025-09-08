import yfinance as yf
import pandas as pd

class YFInterface:
    # Create a global dictionary to store prices
    _prices = {}

    #--- Constructor ---#
    def __init__(self, tickers, start_date, end_date=None):
        """
        Download and cache price data for all tickers between start_date and end_date.
        """
        if not tickers:
            raise ValueError("No tickers found in the database.")

        params = {
            "tickers": tickers,
            "start": start_date,
            "interval": "1d",
            "progress": False,
            "group_by": "ticker",
            "auto_adjust": True
        }

        if end_date is not None:
            params["end"] = end_date

        df = yf.download(**params)


        if isinstance(df.columns, pd.MultiIndex):
            for ticker in tickers:
                self._prices[ticker] = df.xs(ticker, axis=1, level=1)
        else:
            self._prices[tickers[0]] = df  # only one ticker

    
    #--- Function: Get all dates since a given date ---#
    def get_all_dates(self, since_date="2025-09-01"):
        """
        Get all dates since a given date.
        :param since_date: The date to start from in 'YYYY-MM-DD' format.
        :return: A list of dates as strings in 'YYYY-MM-DD' format.
        """
        any_df = next(iter(self._prices.values()))
        data = any_df.index
        # Find index of since_date
        since_index = data.get_loc(since_date, method='pad')
        # Get all dates from since_date to the end of the index
        data = data[since_index:]
        dates = data.strftime('%Y-%m-%d').tolist()
        return dates
    #---------------------------------------------------#

    #--- Function: Check if the market is closed today ---#
    def last_close(self):
        """ Check if the market is closed today by checking if yfinance has a close date for today."""
        any_df = next(iter(self._prices.values()))
        data = any_df.index[-1]  # Get the last date in the index
        date = date.strftime('%Y-%m-%d')
        return date
    #------------------------------------------------------#

    #--- Function: Get the latest close prices for a ticker ---#
    def get_close_prices(self, ticker, start_date, end_date=None):
        """ Get the latest close prices for a ticker from yfinance."""
        if ticker not in self._prices:
            raise ValueError(f"Ticker {ticker} not found in the cached prices.")
        df = self._prices[ticker]
        df = df.loc[start_date:end_date] if end_date else df.loc[start_date:]
        return df['Close'].values
    #------------------------------------------------------#

    #--- Function: Get the latest close prices for a ticker ---#
    def get_price(self, ticker, start_date):
        """Return the closing price for the given date."""
        if ticker not in self._prices:
            raise ValueError(f"Ticker {ticker} not found in the cached prices.")
        
        df = self._prices[ticker]
        # Get index of start_date
        start_index = df.index.get_loc(start_date, method='pad')
        row = df.loc[start_index]
        price = row['Close']
        return price
    #------------------------------------------------------#

    # TODO add function to check ticker validity when front-end is ready