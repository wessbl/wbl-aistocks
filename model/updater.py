from model.db_interface import DBInterface
from model.model import Model

print("*** Beginning Scheduled Update ***")
db = DBInterface()
tickers = db.get_tickers()

# Prepare the day
db.get_day_id('2099-01-01')  # This will ensure the day table is up to date

# TODO 0.7 - Save the closing price

for ticker in tickers:
    print(f"Updating model for {ticker}...")
    model = Model(ticker)
    if model is None:
        # Train the model up to 2025-01-01
        model.train(10, 0.0002)
    model.train(10, 0.0002)
    print(f"Model for {ticker} updated.\n\n")

print("All models updated.")