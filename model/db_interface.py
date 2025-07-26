import sqlite3
import os
from model.yf_interface import YFInterface as yfi
from keras.models import load_model

class DBInterface:
    """Database interface for managing LSTM models and predictions."""
    # Path to saved models/database
    _db_path = None
    _lstm_path = None

    #--- Constructor: Initialize the DBInterface with the path to the database ---#
    def __init__(self, SAVE_PATH):
        # Set the database path
        self._db_path = os.path.join(SAVE_PATH, 'futurestock.db')
        self._lstm_path = SAVE_PATH

        # Verify the database path exists
        if not os.path.exists(self._db_path):
            raise FileNotFoundError(f"Database file not found at {self._db_path}")
        # Verify the LSTM path exists
        if not os.path.exists(self._lstm_path):
            raise FileNotFoundError(f"LSTM path not found at {self._lstm_path}")
    #-----------------------------------------------------------------------------#
    
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            result REAL,
            last_update INTEGER,
            status TEXT,
            version SMALLINT DEFAULT 0
            )
        ''')

        # Store the model in the database
        cursor.execute('''
            INSERT OR REPLACE INTO model (ticker, result, last_update, status)
            VALUES (?, ?, ?, ?)''',
            (ticker, result, last_update, status))
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
            print("Loaded data! Ticker:\t", ticker)
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
                from_day SMALLINT NOT NULL,
                for_day SMALLINT NOT NULL,
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

    #--- Function: Update the Actual Price ---#
    def update_actual_price(self, ticker, for_day, actual_price):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE prediction
            SET actual_price = ?
            WHERE ticker = ? AND for_day = ?''',
            (actual_price, ticker, for_day))
        conn.commit()
        conn.close()
    #---------------------------------------#

    #--- Function: Save Today's Accuracy ---#
    def save_accuracy(self, ticker, day, mape, buy_accuracy, simulated_profit):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_accuracy (
            dailyacc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            day SMALLINT NOT NULL,
            mape REAL,
            buy_accuracy REAL,
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
        
        # If the day does not exist, verify we have all the days leading up to it
        # and save all in the database
        else:
            self._populate_dates()
            cursor.execute('''
                SELECT day_id FROM day WHERE date = ?''',
                (target,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return row[0]
            else:
                return -1
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
            raise ValueError("Day ID not found in the database.")
    #--------------------------------#

    #--- Function: Add all missing dates to the database ---#
    def _populate_dates(self):
        """Populate the database with all dates since the last recorded date."""
        # Get the last day recorded in the database
        today = self.today_id()
        date_list = []
        if today == -1:
            date_list = yfi.get_all_dates()
        else:
            today = self.get_day_string(today)
            date_list = yfi.get_all_dates(since_date=today)
        
        # Add all dates
        for date in date_list:
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