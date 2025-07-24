import yfinance as yf

#--- Function: Get all dates since a given date ---#
def get_all_dates(since_date="2025-01-01"):
    """
    Get all dates since a given date.
    :param since_date: The date to start from in 'YYYY-MM-DD' format.
    :return: A list of dates as strings in 'YYYY-MM-DD' format.
    """
    # Download data from yfinance
    data = yf.download("AAPL", start=since_date, progress=False)
    
    # Check if data is empty
    if data.empty:
        raise ValueError("No data found for the specified date range.")
    
    # Extract dates and convert to string format
    dates = data.index.strftime('%Y-%m-%d').tolist()
    
    return dates
#---------------------------------------------------#

#--- Function: Check if the market is closed today ---#
def last_close():
    """ Check if the market is closed today by checking if yfinance has a close date for today."""

    # Check if yfinance has a close date for today
    df = yf.download("AAPL", period="1d", interval="1d", progress=False)
    if df.empty:
        print("Market has not closed yet today.")                #TODO need to return results
    else:
        print("Market is closed — today's data is available.") #TODO need to return results
#------------------------------------------------------#

#--- Function: Get the latest close prices for a ticker ---#
def get_close_prices(ticker, start_date, end_date=None):
    """"""
    # TODO remove after testing
    # # Get all data from start through last close (yf excludes end date)
    # today = pd.Timestamp.now().date()
    # if today >= self.last_close():
    #     df = yf.download(self.ticker, interval="1d", start=self._start_date)
    # else: 
    #     df = yf.download(self.ticker, start=self._start_date, end=today)

    df = None
    if end_date is None:
        df = yf.download(ticker, interval="1d", start=start_date, progress=False)
    else:
        df = yf.download(ticker, interval="1d", start=start_date, end=end_date, progress=False)
    if (len(df) == 0): raise ValueError("Cannot download from yfinance")

    # Create global orig_data, scaler, scaled_data, X, y
    data = df['Close'].values
    return data
#------------------------------------------------------#
