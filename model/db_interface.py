import sqlite3
import os
from model.model import Model

class DBInterface:
    # Path to database
    base_dir = os.path.dirname(os.path.abspath(__file__))
    _db_path = os.path.join(base_dir, '../static/models/models.db')
    
    # Verify path
    if not os.path.exists(_db_path):
        raise FileNotFoundError(f"Database file not found at {_db_path}")

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
