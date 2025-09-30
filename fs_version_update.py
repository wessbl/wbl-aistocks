import os
import sqlite3
from model.db_interface import DBInterface
from model.yf_interface import YFInterface
from model.model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, 'static', 'models')
old_db = os.path.join(SAVE_PATH, 'models.db')
new_db = os.path.join(SAVE_PATH, 'futurestock.db')

# If you would like to scrub the database, set this to True
SCRUB_DB = True # TODO set to true for release

# Update to 0.7 - db_overhaul
# TODO last changes to 0.7:
    # Model table result should be REAL
    # Front-end should never create a YFInterface
    # daily_accuracy table needs to be updated minimally

def update_fs():
    print("***Updating to version 0.7 - db_overhaul***")

    # Surround with try-except to handle potential errors
    try:
        # Prep: Initialize YFI and DBI
        print("\tPrepping YF and DB interfaces...", end=' ')
        yf = YFInterface(['AAPL'], '2025-09-01')
        dates = yf.get_all_dates()
        db = DBInterface(SAVE_PATH, dates)
        print("done.")

        # Step 1: Rename DB file
        print("\t1. Renaming models.db to futurestock.db...", end=' ')
        try:
            os.rename(old_db, new_db)
            print("done.")
        except FileNotFoundError:
            print("already renamed.")

        # Step 2: Connect to the new DB
        print("\t2. Connecting to the new database...", end=' ')
        conn = sqlite3.connect(new_db)
        cursor = conn.cursor()
        print("done.")

        # Scrub the database (optional)
        if SCRUB_DB:  # Set to True if needed
            print("\tBONUS: Scrubbing the database...", end=' ')
            cursor.execute("DROP TABLE IF EXISTS model;")
            cursor.execute("DROP TABLE IF EXISTS prediction;")
            cursor.execute("DROP TABLE IF EXISTS daily_accuracy;")
            cursor.execute("DROP TABLE IF EXISTS day;")
            conn.commit()
            print("done.")

        # Step 3: Create 'model' table
        print("\t3. Creating model table...", end=' ')
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model';")
        if cursor.fetchone() is not None:
            print("Model table already exists.")
        else:
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
            conn.commit()
            print("done.")
        
        # Step 4: Copy data from old "models" table to new "model" table
        print("\t4. Copying data from old 'models' table to new 'model' table...", end=' ')
        # Check if 'model' already has entries
        cursor.execute("SELECT COUNT() FROM model;")
        count = cursor.fetchone()[0]
        if count > 0:
            print("already done.")
        else:
            try:
                cursor.execute("INSERT INTO model (ticker, blob, status) SELECT ticker, model, status FROM models;")
                conn.commit()
                # Check if the data was copied
                cursor.execute("SELECT COUNT() FROM model;")
                if cursor.fetchone()[0] == 0:
                    print("Error: Couldn't copy.")
                else: print("done.")
            except sqlite3.OperationalError as e:
                print("Error: Couldn't copy.")

        # Step 5: Drop old "models" table
        print("\t5. Dropping old 'models' table...", end=' ')
        cursor.execute("DROP TABLE IF EXISTS models;")
        conn.commit()
        print("done.")

        # Step 6 Create Day table
        print("\t6. Creating Day table if it doesn't exist...", end=' ')
        result = db.get_day_id('2099-01-01')
        if result != -1:
            print("Day table is up to date!")
        else: 
            print("Error: Couldn't find date.")
        
        # Step 7: Create prediction table if it doesn't exist
        print("\t7. Creating prediction table if it doesn't exist...", end=' ')
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
        conn.commit()
        print("done.")

        # Step 8: Create daily_accuracy table if it doesn't exist
        print("\t8. Creating daily_accuracy table if it doesn't exist...", end=' ')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_accuracy (
            dailyacc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            day INTEGER NOT NULL,
            mape REAL,
            buy_accuracy REAL,
            simulated_profit REAL
        )
        ''')
        conn.commit()
        print("done.")

        # Step 9: Add the models if they don't exist
        print("\t9. Adding default models if they don't exist...", end=' ')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        IMG_PATH = os.path.join(BASE_DIR, 'static', 'images')
        models = {}
        if 'AAPL' not in models:
            models['AAPL'] = Model('AAPL', db, yf, IMG_PATH)
        if 'GOOGL' not in models:
            models['GOOGL'] = Model('GOOGL', db, yf, IMG_PATH)
        if 'META' not in models:
            models['META'] = Model('META', db, yf, IMG_PATH)
        if 'AMZN' not in models:
            models['AMZN'] = Model('AMZN', db, yf, IMG_PATH)
        if 'NFLX' not in models:
            models['NFLX'] = Model('NFLX', db, yf, IMG_PATH)
        print("done.")

        # Close the connection
        conn.close()
        print("***Update completed!***")
        return True
    
    except sqlite3.OperationalError as e:
        print(f"An error occurred while updating the database: {e}")
        return False
    except PermissionError as e:
        print(f"Permission error: {e}")
        return False
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return False
    except PermissionError as e:
        print(f"Permission error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False