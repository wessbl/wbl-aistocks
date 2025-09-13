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
yf = YFInterface(tickers, '2025-09-01')
models = []
for ticker in tickers:
    model = Model(ticker, db, yf, IMG_PATH)
    models.append(model)
    if model.get_status() != 'new':
        model.save_actual_price(today, yf.get_price(ticker, db.get_day_string(today)))
print("done.")

# Make sure all actual prices are saved
db.double_check_actual_prices()

# Train models
for model in models:
    print(f"Updater: Updating model for {model.ticker}...")
    model.train(50, 0.01) # TODO set threshold to 0.0002

    print(f"Model for {model.ticker} updated.\n\n")
    # TODO update daily_accuracy table here
    # TODO make sure models are saved correctly

print("Update complete!")