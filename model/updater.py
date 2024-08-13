import schedule
import time
import pytz
from datetime import datetime
from model.db_interface import DBInterface
from model.model import Model

def update_models():
    db = DBInterface()
    tickers = db.get_tickers()
    now = datetime.now(pytz.timezone('US/Eastern'))
    today = now.strftime('%Y-%m-%d')

    for ticker in tickers:
        print(f"Updating model for {ticker}...")
        model = Model(ticker)
        model.train()
        print(f"Model for {ticker} updated.")

    print("All models updated.")

# Schedule the update task
schedule.every().day.at("16:01").do(update_models)

print("Scheduler started. Waiting for the next scheduled update...")
while True:
    schedule.run_pending()
    time.sleep(1)