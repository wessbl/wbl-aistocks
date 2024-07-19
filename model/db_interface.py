import sqlite3

class DBInterface:
    # Path to database
    _db_path = '../static/models/models.db'

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
        cursor.execute('SELECT ticker, last_update FROM models')
        data = cursor.fetchall()
        conn.close()
        pairs = []
        for row in data:
            pairs.append((row[0], row[1]))
        return pairs
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
    dbi.outdate('GOOGL')
    print("Outdated GOOGL")