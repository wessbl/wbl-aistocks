import os
# logging.getLogger('tensorflow').setLevel(logging.ERROR) # Set tf logs to error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Turn off oneDNN custom operations

from model.db_interface import DBInterface
from model.model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(BASE_DIR, '..'))
MODELS_PATH = os.path.join(BASE_DIR, 'static', 'models')
IMG_PATH = os.path.join(BASE_DIR, 'static', 'images')

#TODO double-check that the updater class is running on-time

print("*** Beginning Scheduled Update ***")
db = DBInterface(MODELS_PATH)

# Update days by requesting future date
db.get_day_id('2099-01-01')

# Save actual prices for each ticker
print("Saving actual prices for each ticker...", end=' ')
models = []
tickers = db.get_tickers()
today = db.today_id()
for ticker in tickers:
    model = Model(ticker, MODELS_PATH, IMG_PATH)
    models.append(model)
    model.save_actual_price(today)
print("done.")

# Make sure all actual prices are saved
db.double_check_actual_prices()

# Train models
for model in models:
    print(f"Updating model for {ticker}...")
    model.train(50, 0.0002)
    print(f"Model for {ticker} updated.\n\n")
    # TODO update daily_accuracy table here

print("Update complete!")