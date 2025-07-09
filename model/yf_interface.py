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

