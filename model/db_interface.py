import sqlite3
import os
import pandas as pd
from keras.models import load_model

class DBInterface:
    # Path to database
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _db_path = os.path.join(_base_dir, '../static/models/model.db')
    _lstm_path = os.path.join(_base_dir, '../static/model/')
    
    # Verify path
    if not os.path.exists(_db_path):
        raise FileNotFoundError(f"Database file not found at {_db_path}")
    
    #--- Function: Get path to LSTM file ---#
    def get_lstm_path(self, ticker):
        return os.path.join(self._lstm_path, ticker + '.keras')
    
    #--- Function: Save model to DB ---#
    def save_model(self, ticker, model, last_update, result=''):
        # Save model as file
        path = self.get_lstm_path(ticker)
        model._model.save(path)

        # Read the model file as binary
        with open(path, 'rb') as f:
            model_binary = f.read()

        # Get text version of last_update
        last_update_txt = last_update.strftime("%Y-%m-%d")

        # Database connection
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model (
            id SMALLINT PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            model BLOB,
            result TEXT,
            last_update SMALLINT,
            status TEXT,
            version SMALLINT DEFAULT 0
            )
        ''')

        # Store the model in the database
        cursor.execute('''
            INSERT OR REPLACE INTO model (ticker, model, result, last_update, status)
            VALUES (?, ?, ?, ?, ?)''',
            (ticker, model_binary, result, last_update_txt, 'pending'))    # Pending: front-end needs to be refreshed
        conn.commit()
        conn.close()
    #------------------------------#

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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            day SMALLINT NOT NULL,
            mape REAL,
            buy_accuracy REAL,
            simulated_profit REAL,
            UNIQUE(ticker, date)
        )
        ''')

        # Insert or update the accuracy data
        cursor.execute('''
            INSERT OR REPLACE INTO accuracy (ticker, day, mape, buy_accuracy, simulated_profit)
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
            dayid SMALLINT PRIMARY KEY AUTOINCREMENT,
            date TEXT
            );
        ''')
        cursor.execute('''
            SELECT dayid FROM day WHERE date = ?''',
            (target,))
        row = cursor.fetchone()

        if row:
            conn.close()
            return row[0]  # Return the dayid
        
        # If the day does not exist, verify we have all the days leading up to it
        # and save all in the database
        else:
            self._populate_dates(target)
            cursor.execute('''
                SELECT dayid FROM day WHERE date = ?''',
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
        cursor.execute('SELECT dayid FROM day ORDER BY dayid DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
        else:
            # If no days exist, return -1
            return -1
    #--------------------------------#

    #--- Function: Add all missing dates to the database ---#
    def _populate_dates(self):
        # Get the last day in the database
        last_day = '2025-01-01'
        today = self.today_id()
        if today != -1:
            last_day = today

        # Get all the days since last
        import yfinance as yf
        yf_data = yf.download("AAPL", start=last_day)
        if yf_data.empty:
            raise ValueError("Could not pull yfinance data while trying to create dates.")
        date_list = yf_data.index.tolist()

        # Add all dates
        for date in date_list:
            string_date = date.strftime("%Y-%m-%d")
            self._save_day(string_date)
    #-----------------------------------------------#

    #--- Function: Load from DB ---#
    def load_model(self, ticker):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Handle blob -> model
        cursor.execute('''
                       SELECT model, result, last_update, status
                       FROM model
                       WHERE ticker = ?''',
                       (ticker,))
        row = cursor.fetchone()
        conn.close()

        if row:
            model_data, result, last_update_text, status = row

            # Write model data as .keras file and load it
            path = self.get_lstm_path(ticker)
            with open(path, 'wb') as file:
                file.write(model_data)
            model = load_model(path)

            # Last update txt -> Timestamp
            last_update = pd.Timestamp(last_update_text)

            # Done
            print("Loaded data! Ticker:\t", ticker)
            # print("\nLoaded data! Ticker:\t", ticker,
            #     "\nModel:\t\t", model, "\nLast Update:\t", last_update,
            #     "\nResult:\t\t", result, "\nStatus:\t\t", status)
            return model, result, last_update, status
        else:
            raise ValueError("Model could not be found in the database.")
    #------------------------------#

    #--- Function: drop SQL tables ---#
    def drop_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS model')
        conn.commit()
        conn.close()
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

    #--- Function: Get most data in db ---#
    def get_updated(self):
        #--- Print tickers from the database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT last_update FROM model')
        data = cursor.fetchall()
        conn.close()
        dates = []
        for row in data:
            dates.append(row[0])
        return dates
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

    #--- Function: set stock as outdated ---#
    def outdate(self, ticker):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE model
            SET last_update = ?
            WHERE ticker = ?''',
            ('2024-01-01', ticker))
        conn.commit()
        conn.close()
    #-------------------------------------------#

    #--- Function: Perform Some Update ---#
    def do_version_update(self, instructions):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.executescript(instructions)
        conn.commit()
        conn.close()
    #-------------------------------------#


# TODO Removed this section due to circular import with Model
if __name__ == '__main__':
    # This can be run in the terminal from root directory:
    #       python -m model.db_interface
    # Note that epochs roughly equal minutes for 5 models on the AWS server
    dbi = DBInterface()
    
    entry = -1
    while entry != 0:
        print('''\n*** ADMIN OPERATIONS MENU ***
              1. Outdate all models
              2. Train models - epochs
              3. Train models - MSE Threshold
              0. Exit''')
        entry = input('Please make your selection: ')

        try:
            #   1 - Outdate Models
            entry = int(entry)
            if entry == 1:
                for ticker in dbi.get_tickers():
                    dbi.outdate(ticker)
                print('Outdated all models.')

            #   2 - Train Models by Epoch
            elif entry == 2:
                epochs = int(input('How many epochs? '))
                for ticker in dbi.get_tickers():
                    print('Training model for ' + ticker + '...')
                    model = Model(ticker)
                    model.train(epochs)
                    print('\nFinished training!')
            
            #   3 - Train Models to beat threshold (max 100 epochs)
            elif entry == 3:
                epochs = 50
                threshold = 0.0002
                for ticker in dbi.get_tickers():
                    print('Training model for ' + ticker + 
                          ' until MSE surpasses ' + str(threshold) + '...')
                    model = Model(ticker)
                    model.train(epochs, threshold=threshold)
                    print('\nFinished training!')
            
            #   Finished
            if entry != 0:
                input('Press any key to continue...')

        except ValueError as e:
            print('Invalid entry')
            entry = -1
