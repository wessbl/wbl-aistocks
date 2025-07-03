from model.db_interface import DBInterface as db
from model.model import Model

# Update to 0.7 - db_overhaul
def update_fs():
    # Check if the model table has an int for last_update
    model = Model('AAPL')  # Example ticker, can be any valid ticker
    last_update = model.last_update
    if isinstance(last_update, int):
        print("Already updated to version 0.7 - db_overhaul.")

    # Create Day table
    result = db.get_day_id('2025-07-02')
    if result != -1:
        print("Day table is up to date!")

    
    
    # If not, perform the update
    print('Updating file system to version 0.7 - db_overhaul...')
    instructions = '''
    ALTER TABLE model						
    DROP COLUMN last_update;						
    )'''
    db.do_version_update(instructions)
    instructions = '''
    ALTER TABLE model						
    ADD COLUMN last_update SMALLINT DEFAULT 0;
    )'''
    db.do_version_update(instructions)
    print('File system updated to version 0.7 - db_overhaul successfully!')