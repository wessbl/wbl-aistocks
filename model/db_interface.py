import sqlite3
import os
import numpy as np
from keras.models import load_model

class DBInterface:
    """Database interface for managing LSTM models and predictions."""
    # Path to saved models/database
    _db_path = None
    _lstm_path = None
    _all_dates = None

    #--- Constructor: Initialize the DBInterface with the path to the database ---#
    def __init__(self, SAVE_PATH, all_dates=None):
        # Set the database path
        self._db_path = os.path.join(SAVE_PATH, 'futurestock.db')
        self._lstm_path = SAVE_PATH
        if all_dates is not None:
            self._all_dates = all_dates
            self._populate_dates(self._all_dates) # TODO Can't populate dates until later?

        # Verify the database path exists
        if not os.path.exists(self._db_path):
            raise FileNotFoundError(f"Database file not found at {self._db_path}")
        # Verify the LSTM path exists
        if not os.path.exists(self._lstm_path):
            raise FileNotFoundError(f"LSTM path not found at {self._lstm_path}")
    #-----------------------------------------------------------------------------#

    #--- Function: Check if an updater is already running ---#
    def is_updater_running(self):
        """Check if a model has status in_progress."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM model WHERE status = 'in_progress';
        ''')
        row = cursor.fetchone()
        conn.close()
        if row and row[0] > 0:
            return True
        else:
            return False
    #-----------------------------------------#
    
    #--- Function: Get path to LSTM file ---#
    def get_lstm_path(self, ticker):
        return os.path.join(self._lstm_path, ticker + '.keras')
    
    #--- Function: Save model to DB ---#
    def save_model(self, ticker, model, last_update=None, result='', status='pending'):
        # Save model as file
        path = self.get_lstm_path(ticker)
        model._model.save(path)

        # Database connection
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Create the model table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            result REAL,
            last_update INTEGER,
            status TEXT,
            version INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            UPDATE model
            SET result=?, last_update=?, status=?, version=COALESCE(version, 0) + 1
            WHERE ticker=?
        ''', (result, last_update, status, ticker))

        if cursor.rowcount == 0:  # no row updated, so insert new one
            # Store the model in the database
            cursor.execute('''
                INSERT OR REPLACE INTO model (ticker, result, last_update, status)
                VALUES (?, ?, ?, ?)''',
                (ticker, result, last_update, status))
        conn.commit()
        conn.close()
    #------------------------------#

    #--- Function: Save model accuracy to DB ---#
    def save_model_acc(self, ticker, mape, buy_acc, balance):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE model
            SET mape=?, buy_acc=?, balance=?
            WHERE ticker=?
        ''', (mape, buy_acc, balance, ticker))
        conn.commit()
        conn.close()
    #------------------------------#

    #--- Function: Load from DB ---#
    def load_model(self, ticker):
        # Load the model from file
        path = self.get_lstm_path(ticker)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        model = load_model(path, compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Get model data from the database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT result, last_update, status
                       FROM model
                       WHERE ticker = ?''',
                       (ticker,))
        row = cursor.fetchone()
        conn.close()

        if row:
            result, last_update, status = row
            print("Loaded data for ticker: ", ticker)
            # print("\nLoaded data! Ticker:\t", ticker,
            #     "\nModel:\t\t", model, "\nLast Update:\t", last_update,
            #     "\nResult:\t\t", result, "\nStatus:\t\t", status)
            return model, result, last_update, status
        else:
            raise ValueError("Model could not be found in the database.")
    #------------------------------#

    #--- Function: Get all the tickers in db ---#
    def get_tickers(self):
        #--- Print tickers from the database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT ticker FROM model')
        data = cursor.fetchall()
        conn.close()
        tickers = []
        for row in data:
            tickers.append(row[0])
        return tickers
    #-------------------------------------------#

    #--- Function: Change the status ---#
    def set_status(self, ticker, status):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE model
            SET status = ?
            WHERE ticker = ?''',
            (status, ticker))
        conn.commit()
        conn.close()
    #-----------------------------------#

    #--- Function: Check the status ---#
    def get_status(self, ticker):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT status FROM model WHERE ticker = ?', (ticker,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]   # Return the status as a string
        else:
            raise ValueError("Ticker not found in the database.")
    #-----------------------------------#

    #--- Function: Save Day to DB ---#
    def _save_day(self, day_string):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Insert or update the day data
        cursor.execute('''
            INSERT INTO day (date)
            VALUES (?)''',
            (day_string,))
        conn.commit()
        conn.close()
    #--------------------------------#

    #--- Function: Save Prediction to DB ---#
    def save_prediction(self, ticker, from_day, for_day, predicted_price, buy):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction (
                predict_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                from_day INTEGER NOT NULL,
                for_day INTEGER NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                buy BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        # Insert or update the prediction data
        cursor.execute('''
            INSERT OR REPLACE INTO prediction (ticker, from_day, for_day, predicted_price, actual_price, buy)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (ticker, from_day, for_day, predicted_price, None, buy))
        conn.commit()
        conn.close()
    #---------------------------------------#

    #--- Function: Get Predictions from DB ---#
    def get_predictions(self, ticker, end_day):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ticker, from_day, for_day, predicted_price, actual_price, buy
            FROM prediction
            WHERE ticker = ? AND for_day = ?
            ORDER BY for_day ASC
        ''', (ticker, end_day))
        rows = cursor.fetchall()
        conn.close()

        # Return as data frame
        import pandas as pd
        if rows:
            df = pd.DataFrame(
                rows, columns=['ticker', 'from_day', 'for_day', 'predicted_price', 'actual_price', 'buy']
            )
            return df
            
        else:
            raise ValueError("No predictions found.")
    #---------------------------------------#

    #--- Function: Update the Actual Price ---#
    def save_actual_price(self, ticker, for_day, price):
        """Update the actual price in prediction table for a given ticker and day."""
        if np.isnan(price):
            raise ValueError(f"Price for {ticker} on day {for_day} is NaN, cannot save actual price.")
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE prediction
            SET actual_price = ?
            WHERE ticker = ? AND for_day = ?''',
            (price, ticker, for_day))
        conn.commit()
        if cursor.rowcount == 0:
            print(f"Warning: No prediction found for {ticker} on day {for_day}. Actual price not updated.")
        elif cursor.rowcount > 5:
            print(f"Warning: Updated actual_price for {cursor.rowcount} rows: {ticker} on day {for_day}.")
        # TODO remove after testing
        # else:
        #     print(f"Saved actual_price to {cursor.rowcount} row(s): {ticker} on day {for_day}.")
        conn.close()
    #---------------------------------------#

    #--- Function: Double-check actual prices ---#
    def double_check_actual_prices(self, today):
        """Double-check that all actual prices are saved in the database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ticker, for_day FROM prediction
            WHERE actual_price IS NULL AND for_day < ?''',
            (today,))
        rows = cursor.fetchall()
        conn.close()
        
        # Create a dictionary [tickers] -> [missing days]
        missing_data = {}
        for ticker, for_day in rows:
            missing_data[ticker] = missing_data.get(ticker, []) + [for_day]

        return missing_data
    #---------------------------------------#

    #--- Function: Get Max Buy Accuracy ---#
    def get_buy_accuracy(self, ticker, return_day=False):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MAX(buy_accuracy), day
            FROM daily_accuracy
            WHERE ticker = ?
                AND buy_accuracy IS NOT NULL''',
            (ticker,))
        row = cursor.fetchall()
        conn.close()

        if return_day:
            if row and row[0][0] is not None:
                return row[0][0], row[0][1]  # Return the max buy_accuracy and the day it occurred
            else:
                return 0, 1 # First day has 0 predictions on day 1
            
        else:
            if row and row[0][0] is not None:
                return row[0][0]  # Return the max buy_accuracy
            else:
                return 0 # First day has 0 predictions on day 1
    #---------------------------------------#

    # TODO remove after testing
    #--- Function: Get Max Buy Accuracy Average ---#
    # def get_buy_acc_avg(self, ticker):
    #     conn = sqlite3.connect(self._db_path)
    #     cursor = conn.cursor()
    #     cursor.execute('''
    #         SELECT buy_accuracy FROM daily_accuracy
    #         WHERE ticker = ? AND buy_accuracy IS NOT NULL''',
    #         (ticker,))
    #     row = cursor.fetchone()
    #     conn.close()
    #     if row and row[0] is not None:
    #         # Return the avg buy_accuracy
    #         avg = max(row) / len(row)
    #         return round(avg, 2)
    #     else:
    #         return 0
    #---------------------------------------#

    #--- Function: Get All MAPE values ---#
    def get_mape(self, ticker):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT mape FROM daily_accuracy
            WHERE ticker = ? AND mape IS NOT NULL''',
            (ticker,))
        rows = cursor.fetchall()
        conn.close()
        if rows:
            return [row[0] for row in rows]  # Return list of MAPE values
        else:
            return []
    #---------------------------------------#

    #--- Function: Get Simulated Profit ---#
    def get_simulated_profit(self, ticker, day):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT simulated_profit FROM daily_accuracy
            WHERE ticker = ? AND day = ?''',
            (ticker, day))
        row = cursor.fetchone()
        conn.close()
        if row and row[0] is not None:
            return row[0]   # Return the simulated_profit
        else:
            return 1000.0  # Default starting profit
    #---------------------------------------#

    #--- Function: Save Today's Accuracy ---#
    def save_accuracy(self, ticker, day, mape, buy_accuracy, simulated_profit):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_accuracy (
            dailyacc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            day INTEGER NOT NULL,
            mape REAL,
            buy_accuracy INTEGER,
            simulated_profit REAL,
            UNIQUE(ticker, day)
        )
        ''')

        # Insert or update the accuracy data
        cursor.execute('''
            INSERT OR REPLACE INTO daily_accuracy (ticker, day, mape, buy_accuracy, simulated_profit)
            VALUES (?, ?, ?, ?, ?)''',
            (ticker, day, mape, buy_accuracy, simulated_profit))
        conn.commit()
        conn.close()
    #---------------------------------------#

    #--- Function: Find entries in daily_accuracy with NULL values ---#
    def daily_acc_empty_cells(self, tickers, today):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        for ticker in tickers:
            # Most of the time, only today's entry will be missing
            cursor.execute('''
                    SELECT COUNT(*) FROM daily_accuracy
                    WHERE ticker = ?''',
                    (ticker,))
            count = cursor.fetchone()[0]
            if count >= today - 1:
                # Insert today's entry with NULL values
                cursor.execute('''
                    INSERT OR IGNORE INTO daily_accuracy (ticker, day)
                    VALUES (?, ?)''',
                    (ticker, today))
                conn.commit()
            
            else:
                # Insert any missing days with NULL values
                print(f"Ticker {ticker} has {count} entries, expected {today - 1}. Adding missing days...", end=' ')
                for day in range(1, today):
                    # Check if it's already in the database
                    cursor.execute('''
                        SELECT COUNT(*) FROM daily_accuracy
                        WHERE ticker = ? AND day = ?''',
                        (ticker, day))
                    count = cursor.fetchone()[0]
                    if count == 0:
                        # Insert the missing day with NULL values
                        cursor.execute('''
                            INSERT OR IGNORE INTO daily_accuracy (ticker, day)
                            VALUES (?, ?)''',
                            (ticker, day))
                        conn.commit()
                print("done.")
        # Return all entries with NULL values
        cursor.execute('''
            SELECT ticker, day FROM daily_accuracy
            WHERE simulated_profit IS NULL
        ''')
        rows = cursor.fetchall()
        conn.close()

        # Create a dictionary [tickers] -> [missing days]
        missing_data = {}
        for ticker, day in rows:
            missing_data[ticker] = missing_data.get(ticker, []) + [day]

        return missing_data
    #---------------------------------------#
    
    #--- Function: Get the integer ID of the day ---#
    def get_day_id(self, target):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS day (
            day_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT
            );
        ''')
        cursor.execute('''
            SELECT day_id FROM day WHERE date = ?''',
            (target,))
        row = cursor.fetchone()

        if row:
            conn.close()
            return row[0]  # Return the day_id
        
        # If the day does not exist, try to update the table with all dates
        else:
            if self._all_dates is None:
                raise ValueError("Day not found in the database. Please provide all_dates to populate missing dates.")
            self._populate_dates(self._all_dates)
            cursor.execute('''
                SELECT day_id FROM day WHERE date = ?''',
                (target,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return row[0]  # Return the day_id
            else:
                return -1  # If still not found, return -1
    #--------------------------------#

    #--- Function: Get today's day ID ---#
    def today_id(self):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT day_id FROM day ORDER BY day_id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
        else:
            # If no days exist, return -1
            return -1
    #--------------------------------#

    #--- Function: Get string from day id ---#
    def get_day_string(self, day_id):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT date FROM day WHERE day_id = ?', (day_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
        else:
            ""
    #--------------------------------#

    #--- Function: Get all dates in the database ---#
    def all_dates(self):
        """Get the total number of days in the database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT date FROM day')
        row = cursor.fetchall()
        conn.close()
        if row:
            row = [r[0] for r in row]  # Extract the date strings from the tuples
            return row
        else:
            raise ValueError("No days found in the database.")
    #----------------------------------------------#

    #--- Function: Get all days in the database ---#
    def all_days (self):
        """Get the total number of days in the database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT day_id FROM day')
        row = cursor.fetchall()
        conn.close()
        if row:
            row = [r[0] for r in row]  # Extract the date strings from the tuples
            return row
        else:
            raise ValueError("No days found in the database.")
    #----------------------------------------------#


    #--- Function: Add all missing dates to the database ---#
    def _populate_dates(self, dates):
        """Populate the database with all dates since the last recorded date."""
        # Get the last day recorded in the database
        today = self.get_day_string(self.today_id())
        if today != "" and today in dates:
            # find position of 'today' in the list
            index = dates.index(today)
            dates = dates[index+1:] # Remove the first date since it's already in the database

        for date in dates:
            self._save_day(date)
    #-----------------------------------------------#

    #--- Function: Perform Some Update ---#
    def do_update(self, instructions):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.executescript(instructions)
        conn.commit()
        conn.close()
    #-------------------------------------#

    #--- Function: Perform Some Query ---#
    def run_query(self, instructions):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(instructions)
        result = cursor.fetchall()
        conn.close()
        return result
    #-------------------------------------#
