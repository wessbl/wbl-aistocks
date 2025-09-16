import os
# logging.getLogger('tensorflow').setLevel(logging.ERROR) # Set tf logs to error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Turn off oneDNN custom operations

from model.db_interface import DBInterface
from model.yf_interface import YFInterface
from model.model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(BASE_DIR, '..'))
MODELS_PATH = os.path.join(BASE_DIR, 'static', 'models')
IMG_PATH = os.path.join(BASE_DIR, 'static', 'images')

#TODO the updater class is not running on time, check logs

print("*** Beginning Scheduled Update ***")
# Instantiate classes
db = DBInterface(MODELS_PATH)
tickers = db.get_tickers()
today = db.today_id()
yf = YFInterface(tickers, '2017-01-01')
models = []
for ticker in tickers:
    model = Model(ticker, db, yf, IMG_PATH)
    models.append(model)
    # TODO I don't think this  is needed, actual_price saved elsewhere
    # if model.get_status() != 'new':
    #     model.save_actual_price(today, yf.get_price(ticker, db.get_day_string(today))) # TODO is this needed?
print("done.")

# Make sure all actual prices are saved
missing = db.double_check_actual_prices(today)
if missing:
    print("WARNING: Some actual prices are still missing. Attempting to save...")
    for ticker, days in missing.items():
        for day in days:
            try:
                print(f"\tWARNING: Actual price for {ticker} on day {day} is missing. Attempting to save...")
                db.save_actual_price(ticker, day, yf.get_price(ticker, db.get_day_string(day)))
                print("saved successfully!")
            except ValueError as e:
                print(f"Error saving actual price for {ticker} on {day}: {e}")
    print("Done checking actual prices.")
else:
    print("All actual prices are saved.")

# Train models
for model in models:
    print(f"Updater: Updating model for {model.ticker}...")
    try:
        model.train(50, 0.01) # TODO set threshold to 0.0002
    except ValueError as e:
        print(f"Error updating model for {model.ticker}: {e}")
        continue

    print(f"Model for {model.ticker} updated.\n\n")
    # TODO update daily_accuracy table here
    # TODO make sure models are saved correctly

print("Update complete!")