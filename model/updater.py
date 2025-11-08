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

# Instantiate classes and key variables
error_occurred = False
db = DBInterface(MODELS_PATH)
tickers = db.get_tickers()
yf = YFInterface(tickers, '2017-01-01')
db.populate_dates(yf.get_all_dates()) # Ensure dates table is populated
db.prepare_daily_acc(tickers)  # Add new dates
today = db.today_num()
models = []
for ticker in tickers:
    model = Model(ticker, db, yf, IMG_PATH)
    models.append(model)

# Set all tickers to 'completed' status in case you killed the updater halfway through
for ticker in tickers:
    db.set_status(ticker, 'completed')

# Check if an updater is already running
if db.is_updater_running():
    print("Another updater instance is already running. Exiting.")
    exit()
# Set the first ticker to 'in_progress' to indicate updater is running
db.set_status(tickers[0], 'in_progress')

# Make sure all actual prices are saved
missing = db.double_check_actual_prices(today)
if missing:
    print("WARNING: Some actual prices are still missing. Attempting to save...")
    for ticker, days in missing.items():
        for day in days:
            try:
                # print(f"\tWARNING: Actual price for {ticker} on day {day} is missing. Attempting to save...")
                db.save_actual_price(ticker, day, yf.get_price(ticker, db.get_day_string(day)))
                # print("saved successfully!")
            except ValueError as e:
                error_occurred = True
                print(f"\tError saving actual price for {ticker} on {day}: {e}")
    print("Done checking actual prices.")
else:
    print("All actual prices are saved.")

# Train models and calculate daily accuracy
print()
last_model = None # Keep track of last model so there's always one set to in_progress
for model in models:
    print(f"Updater: Training model for {model.ticker}...")
    try:
        # Manage status, which acts as a lock to prevent multiple updaters running simultaneously
        new = False
        if model._lstm.status == 'new':
            new = True
        model.set_status(2) # in_progress 
        if last_model is not None:
            last_model.set_status(3)
        
        # Train every day since last update
        # TODO 0.9 do initial training since the LSTM's start date (2017-01-01)
        # TODO 0.7 set threshold to 0.0002
        model.train(epochs=1)
        
        # Calculate Daily Accuracy for any missing days
        ticker = model.ticker
        blank_entries = db.daily_acc_empty_cells(ticker)
        if not blank_entries or len(blank_entries) == 0:
            print(f"{ticker} already has all daily accuracy calculations completed.")
            last_model = model
            continue

        # For each day that is missing for this ticker, calculate and save the daily accuracy
        else:
            print(f"Calculating daily accuracy for {ticker}...")
            for day in blank_entries:
                mape = None
                ape = None
                buy_acc = None
                balance = 100.0 # Start with $100 (which also means 100%)

                # Save generic first day values
                if day == 1:
                    print(f"\nDay {day}:") # TODO remove
                    db.save_accuracy(ticker, day, ape, mape, buy_acc, balance)
                    print("\t1: Saved day 1 accuracy") # TODO remove
                # Calculate values since previous day
                else:
                    print(f"\nDay {day}:") # TODO remove
                    # Get all predictions up to today
                    df = db.get_predictions(ticker, day) # why is the ape Nan for AAPL and None for AMZN??
                    print("\t1: Got predictions") # TODO remove
                    # print(f"\t\tdf = \n{df}") # TODO remove
                    
                    # Calculate today's Absolute Percentage Errors (APE) and store them
                    ape_df = df[df['ape'].isna()]
                    ape_df = ape_df[ape_df['for_day'] <= day]
                    # print(f"\t\tape_df before calc = \n{ape_df}") # TODO remove
                    ape_df['ape'] = abs((ape_df['actual_price'] - ape_df['predicted_price']) / ape_df['actual_price']) * 100
                    # Create dictionary of ids -> newly calculated apes
                    # print(f"\t\tape_df after calc = \n{ape_df}") # TODO remove
                    today_apes = ape_df[['predict_id', 'ape']].to_dict(orient='records')
                    # print(f"\t\ttoday_apes = \n{today_apes}") # TODO remove
                    for entry in today_apes:
                        id = entry['predict_id']
                        ape = entry['ape']
                        # print(f"\t\tSaving APE at prediction ID {id}: {ape}") # TODO remove
                        db.save_ape(id, ape)
                    print("\t2: Calculated and saved APEs") # TODO remove
                    
                    # Calculate Mean Absolute Percentage Error (MAPE) up to today
                    apes = db.get_apes(ticker, day) # Refresh df to include newly saved apes
                    mape = sum(apes) / len(apes)
                    mape = round(mape, 2)
                    print(f"\t3: Calculated MAPE: {mape}") # TODO remove

                    # Calculate buy accuracy
                    today_price = yf.get_price(ticker, db.get_day_string(day))
                    yesterday = day - 1
                    yesterday_price = yf.get_price(ticker, db.get_day_string(yesterday))
                    stock_went_up = today_price > yesterday_price
                    print("\t4: Determined stock movement") # TODO remove

                    # Get yesterday's buy prediction for today
                    # print(f"\t\tdf = \n{df}")
                    row = df.loc[(df['from_day'] == yesterday)]
                    # print(f"\t\trow = \n{row}") # TODO remove
                    yesterday_buy = row['buy'].iloc[0] if not row.empty else None
                    # print(f"\t5: yesterday_buy raw value: {yesterday_buy}") # TODO remove
                    buy_acc = db.get_buy_accuracy(ticker)
                    if yesterday_buy == stock_went_up:
                        buy_acc += 1
                    print("\t6: Calculated buy accuracy") # TODO remove

                    # Calculate simulated profit
                    balance = db.get_simulated_profit(ticker, yesterday)
                    if yesterday_buy: # If the model recommended buying yesterday
                        percentage = (today_price - yesterday_price) / yesterday_price
                        profit = balance * percentage
                        balance += profit
                        balance = round(balance, 2)
                    print("\t7: Calculated simulated profit") # TODO remove
                    
                    # Debug Prints
                    if yesterday_buy == stock_went_up:
                        if stock_went_up:
                            print(f"\tGOOD: Made a profit of ${profit}! New balance: ${balance}.")
                        else:
                            # this is a repetitive calculation, optimize if you want to keep these prints
                            percentage = (today_price - yesterday_price) / yesterday_price
                            profit = balance * percentage
                            profit = round(profit, 2)
                            print(f"\tGOOD: Avoided a loss of ${-profit}! Balance remains: ${balance}.")
                    else:
                        if stock_went_up:
                            # this is a repetitive calculation, optimize if you want to keep these prints
                            percentage = (today_price - yesterday_price) / yesterday_price
                            profit = balance * percentage
                            profit = round(profit, 2)
                            print(f"\tFAIL: Missed a profit of ${profit}. Balance remains: ${balance}.")
                        else:
                            print(f"\tFAIL: Incurred a loss of ${-profit}. New balance: ${balance}.")

                    
                    # Save to DB
                    print(f"\tDay {day}: MAPE: {mape}, Buy Accuracy: {buy_acc}, Balance: {balance}")
                    db.save_accuracy(ticker, day, ape, mape, buy_acc, balance)

            # Calculate all-time MAPE; get previous values as well as today's
            print(f"Calculating and saving to model table...")
            mape = db.get_mape(ticker, day)

            # Calculate the model's all-time buy accuracy
            max_acc = db.get_buy_accuracy(ticker)
            all_time_acc = max_acc * 100 / (today - 1) # Exclude day 1 since no prediction was made for it
            all_time_acc = round(all_time_acc, 2)

            # Save all_time data to DB
            print(f"\tMAPE: {mape}, Accuracy: {all_time_acc}, Balance: {balance}")
            db.save_model_acc(ticker, mape, all_time_acc, balance)
            print("done.")

        # TODO 0.7 Verify MAPE calculation correctness. Wrong because it's calculating an
        #   average of averages, track the APE values individually instead.
        # TODO 0.7 Verify buy accuracy calculation correctness as fetching has been changed.
        # TODO 0.7 Verify balance calculation correctness
        last_model = model

    except ValueError as e:
        error_occurred = True
        print(f"ValueError updating model for {model.ticker}: {e}")
        print(yf.get_close_prices(model.ticker, '2017-01-01'))
        continue

    print(f"Model for {model.ticker} updated.\n")

# Wrap up updates
if last_model is not None:
    last_model.set_status(3) # completed
if error_occurred:
    erroneous_tickers = db.finish_update()
    print(f"Errors occurred on tickers {erroneous_tickers}.")
else:
    print("No errors occurred!")

# TODO 0.8 It might be nice to have the updater do a once-over of data on the weekends
print("***Update complete!***")