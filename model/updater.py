import os
import pandas as pd
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

print("*** Beginning Scheduled Update ***")

# Check if an updater is already running
db = DBInterface(MODELS_PATH)
if db.is_updater_running():
    print("Another updater instance is already running. Exiting.")
    exit()

# Instantiate classes
tickers = db.get_tickers()
today = db.today_id()
yf = YFInterface(tickers, '2017-01-01')
models = []
for ticker in tickers:
    model = Model(ticker, db, yf, IMG_PATH)
    models.append(model)

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
        # TODO train every day since last update
        model.train(50, 0.01) # TODO set threshold to 0.0002
    except ValueError as e:
        print(f"Error updating model for {model.ticker}: {e}")
        print(yf.get_close_prices(model.ticker, '2017-01-01'))
        continue

    print(f"Model for {model.ticker} updated.\n\n")

# Calculate Daily Accuracy
print("Calculating daily accuracy for all models...")
db.do_update("DELETE FROM daily_accuracy;") # TODO remove after testing
blank_entries = {}
blank_entries = db.daily_acc_empty_cells(tickers, today) # DB will update the table before returning the blank entries
if blank_entries:
    for ticker, days in blank_entries.items():
        for day in days:
            mape = None
            buy_acc = None
            balance = 100.0 # Start with $100
            try:
                # Save generic first day values
                if day == 1:
                    db.save_accuracy(ticker, day, mape, buy_acc, balance)
                # Calculate values since previous day
                else:
                    # Calculate today's Mean Absolute Percentage Error (MAPE)
                    df = db.get_predictions(ticker, day)
                    df['error'] = abs((df['actual_price'] - df['predicted_price']) / df['actual_price'])
                    # print(df) # TODO remove after testing
                    # Handle division by zero just in case
                    df = df[df['actual_price'] != 0]
                    mape = df['error'].mean() * 100
                    mape = round(mape, 2)

                    # Calculate buy accuracy
                    today_price = yf.get_price(ticker, db.get_day_string(day))
                    yesterday = day - 1
                    yesterday_price = yf.get_price(ticker, db.get_day_string(yesterday))
                    stock_went_up = today_price > yesterday_price

                    # Get yesterday's predictions
                    mask = df['from_day'] == yesterday
                    yesterday_pred = df[mask]

                    # Find if buy was true for yesterday's predictions
                    yesterday_buy = yesterday_pred['buy'].iloc[0]
                    buy_acc = db.get_buy_accuracy(ticker) # TODO buy_acc is None
                    if yesterday_buy == stock_went_up: 
                        buy_acc += 1

                    # Calculate simulated profit
                    balance = db.get_simulated_profit(ticker, yesterday)
                    if yesterday_pred['buy'].iloc[0]: # If the model recommended buying yesterday
                        if stock_went_up:
                            percentage = (today_price - yesterday_price) / yesterday_price
                            profit = balance * percentage
                            profit = round(profit, 2)
                            balance += profit
                    
                    # Save to DB
                    print(f"Saving daily accuracy for {ticker} on day {day}...", end=' ')
                    db.save_accuracy(ticker, day, mape, buy_acc, balance)
                    print("saved.")

                    # Calculate all-time MAPE; get previous values as well as today's
                    print(f"Calculating and saving to model table...", end=' ')
                    mape_list = db.get_mape(ticker)
                    mape_list.append(mape)
                    all_time_mape = sum(mape_list) / len(mape_list)
                    all_time_mape = round(all_time_mape, 2)

                    # Calculate the model's all-time buy accuracy
                    max_acc, length = db.get_buy_accuracy(ticker, return_day=True)
                    all_time_acc = 0.0
                    if length > 1:
                        length -= 1 # Subtract 1 to not count the first day
                        all_time_acc = max_acc * 100 / length
                        all_time_acc = round(all_time_acc, 2)

                    # Save all_time data to DB
                    db.save_model_acc(ticker, all_time_mape, all_time_acc, balance)
                    print("done.")
                    
            except ValueError as e:
                print(f"Error calculating daily accuracy for {ticker} on {day}: {e}")
    print("Done calculating daily accuracy.")

print("***Update complete!***")