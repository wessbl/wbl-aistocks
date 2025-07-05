import os
import sqlite3
from model.db_interface import DBInterface as db
from model.model import Model

old_db = 'static/models/models.db'
new_db = 'static/models/futurestock.db'


# Update to 0.7 - db_overhaul
def update_fs():
    # Step 1: Rename DB file if needed
    if not os.path.exists(old_db):
        print("Already updated to version 0.7 - db_overhaul.")
        return
    
    print("***Updating to version 0.7 - db_overhaul***")

    # Surround with try-except to handle potential errors
    try:
        # Step 1: Rename DB file
        print("1. Renaming models.db to futurestock.db...", end=' ')
        os.rename(old_db, new_db)
        print("done.")

        # Step 2: Connect to the new DB
        print("2. Connecting to the new database...", end=' ')
        conn = sqlite3.connect(new_db)
        cursor = conn.cursor()
        print("done.")

        # Step 3: Rename table if it's still named 'models'
        print("3. Renaming table 'models' to 'model'...", end=' ')
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        if 'models' in tables and 'model' not in tables:
            cursor.execute("ALTER TABLE models RENAME TO model;")
            conn.commit()
            print("done.")
        else:
            print("rename already done.")
        
        # Step 4: Add 'id' and 'version' columns in model table
        print("4. Update model table:")
        cursor.execute("PRAGMA table_info(model);")
        columns = [row[1] for row in cursor.fetchall()]
        if 'id' in columns and 'version' in columns:
            print("columns already exist.")
        else:
            # Add 'id' column
            print("\tAdding 'id' column...", end=' ')
            cursor.execute("ALTER TABLE model ADD COLUMN id INTEGER PRIMARY KEY AUTOINCREMENT;")
            conn.commit()
            print("done.")

            # Add 'version' column
            print("\tAdding 'version' column...", end=' ')
            cursor.execute("ALTER TABLE model ADD COLUMN version SMALLINT DEFAULT 0;")
            conn.commit()
            print("done.")

            # Drop 'last_update' column if it exists
            print("\tDropping 'last_update' column if it exists...", end=' ')
            cursor.execute("ALTER TABLE model DROP COLUMN last_update;")
            conn.commit()
            print("done.")

            # Add 'last_update' column
            print("\tAdding 'last_update' column...", end=' ')
            cursor.execute("ALTER TABLE model ADD COLUMN last_update SMALLINT DEFAULT 0;")
            conn.commit()
            print("done.")

        # Create Day table
        result = db.get_day_id('2025-07-02')
        if result != -1:
            print("Day table is up to date!")
        
        

    except sqlite3.OperationalError as e:
        print(f"An error occurred while updating the database: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")