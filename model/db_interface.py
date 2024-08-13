import sqlite3
import os
from model.model import Model

class DBInterface:
    # Path to database
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _db_path = os.path.join(base_dir, '../static/models/models.db')
    
    # Verify path
    if not os.path.exists(_db_path):
        raise FileNotFoundError(f"Database file not found at {_db_path}")
    
    #--- Function: Save to DB ---#
    def _save(self, ticker, model, last_update, result=''):
        # Save model as file
        model._lstm.save(self._lstm_path)

        # Read the model file as binary
        with open(self._lstm_path, 'rb') as f:
            model_binary = f.read()

        # Get text version of last_update
        last_update_txt = last_update.strftime("%Y-%m-%d")

        # Database connection
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
            ticker TEXT PRIMARY KEY,
            model BLOB,
            result TEXT
            last_update TEXT,
            status TEXT
            )
        ''')    # TODO ENSURE ALL DB INTERACTIONS FOLLOW NEW LAYOUT ABOVE!

        # Store the model in the database
        cursor.execute('''
            INSERT OR REPLACE INTO models (ticker, model, result, last_update, status)
            VALUES (?, ?, ?, ?)''',
            (ticker, model, result, last_update, 'pending'))    # Pending: front-end needs to be refreshed
        conn.commit()
        conn.close()
    #------------------------------#

    #--- Function: Load from DB ---#
    def _load(self, ticker):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Handle blob -> model
        cursor.execute("SELECT * FROM models WHERE ticker = " + ticker)
        row = cursor.fetchone()[0]
        conn.close()

        if row:
            last_update_text, result = row

            # Last update txt -> Timestamp
            last_update = pd.Timestamp(last_update_text)

            # Done
            print("\nLoaded data from database!\nTicker:\t\t", ticker,
                "\nModel:\t\t", model, "\nLast Update:\t", last_update,
                "\nResult:\t\t", result)
            return model, result, last_update, status
        else:
            raise ValueError("Model could not be found in the database.")

        # if row:
        #     result, status = row
        #     # Done
        #     print("\nModel for "+ticker+" loaded from database...")
        #     # print("\nLoaded data from database!\nTicker:\t\t", ticker,
        #     #     "\nModel:\t\t", model, "\nLast Update:\t", last_update,
        #     #     "\nResult:\t\t", result)
        #     return model, last_update, result

        # else:
        #     raise ValueError("Model could not be found in the database.")
    #------------------------------#

    #--- Function: drop SQL tables ---#
    def drop_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS models')
        conn.commit()
        conn.close()
    #------------------------------#

    #--- Function: Get all the tickers in db ---#
    def get_tickers(self):
        #--- Print tickers from the database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT ticker FROM models')
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
        cursor.execute('SELECT last_update FROM models')
        data = cursor.fetchall()
        conn.close()
        dates = []
        for row in data:
            dates.append(row[0])
        return dates
    #-------------------------------------------#

    #--- Function: set stock as outdated ---#
    def outdate(self, ticker):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE models 
            SET last_update = ? 
            WHERE ticker = ?''',
            ('2024-01-01', ticker))
        conn.commit()
        conn.close()
    #-------------------------------------------#


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
