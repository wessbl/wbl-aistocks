import sqlite3
import os

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
    dbi = DBInterface()
    print("Tickers:\t", dbi.get_tickers())
    print("Updated:\t", dbi.get_updated())
    for tick in dbi.get_tickers():
        dbi.outdate(tick)
    print("Outdated all models.")
    print("Updated:\t", dbi.get_updated())
