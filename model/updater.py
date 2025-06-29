from model.db_interface import DBInterface
from model.model import Model


print("*** Beginning Scheduled Update ***")
db = DBInterface()
tickers = db.get_tickers()

for ticker in tickers:
    print(f"Updating model for {ticker}...")
    model = Model(ticker)
    model.train(10, 0.0002)
    print(f"Model for {ticker} updated.\n\n")

print("All models updated.")