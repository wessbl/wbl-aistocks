import schedule
import time
import pytz
from datetime import datetime, timedelta
from model.db_interface import DBInterface
from model.model import Model

# The time to update, gives 5 extra minutes to ensure API
# has correct closing price
update_time = "16:05"

# Debug: set update time to 1 minute from now
now = datetime.now(pytz.timezone('US/Eastern'))
now = now + timedelta(minutes=1)
update_time = now.strftime("%H:%M")

def update_models():
    print("*** Beginning Scheduled Update ***")
    db = DBInterface()
    tickers = db.get_tickers()

    for ticker in tickers:
        print(f"Updating model for {ticker}...")
        model = Model(ticker)
        model.train()
        print(f"Model for {ticker} updated.")

    print("All models updated.")

def get_eastern_time():
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern).strftime("%H:%M")

print("Scheduler started. Waiting for the next scheduled update...")
while True:
    if get_eastern_time() == update_time:
        update_models()
    else: print("Waiting for " + update_time + ", it's currently " +get_eastern_time()+ ".")   #TODO remove debug print

    time.sleep(15)