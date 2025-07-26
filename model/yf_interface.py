import yfinance as yf

class YFInterface:
    #--- Function: Get all dates since a given date ---#
    def get_all_dates(since_date="2025-01-01"):
        """
        Get all dates since a given date.
        :param since_date: The date to start from in 'YYYY-MM-DD' format.
        :return: A list of dates as strings in 'YYYY-MM-DD' format.
        """
        # Download data from yfinance
        data = yf.download("AAPL", period="1d", start=since_date, progress=False, auto_adjust=True)
        
        # Extract dates and convert to string format
        dates = data.index.strftime('%Y-%m-%d').tolist()
        
        return dates
    #---------------------------------------------------#

    #--- Function: Check if the market is closed today ---#
    def last_close():
        """ Check if the market is closed today by checking if yfinance has a close date for today."""

        # Check if yfinance has a close date for today
        df = yf.download("AAPL", period="1d", interval="1d", progress=False, auto_adjust=True)

        # TODO test to make sure it returns the yesterday's date if the market is closed today

        return df.index[-1].strftime('%Y-%m-%d')
    #------------------------------------------------------#

    #--- Function: Get the latest close prices for a ticker ---#
    def get_close_prices(ticker, start_date, end_date=None):
        """ Get the latest close prices for a ticker from yfinance."""
        df = None
        if end_date is None:
            df = yf.download(ticker, interval="1d", start=start_date, progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, interval="1d", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if (len(df) == 0): raise ValueError("Cannot download from yfinance")

        # Create global orig_data, scaler, scaled_data, X, y
        data = df['Close'].values
        return data
    #------------------------------------------------------#
