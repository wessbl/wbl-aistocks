from flask import Flask, render_template, request, jsonify, send_from_directory
from model.model import Model
import os
import time

# Dictionary of models ticker : Model
models = {}

app = Flask(__name__)

# Show /index.html
@app.route('/')
@app.route('/stocks')
def home():
    return render_template('index.html')

# White paper download
@app.route('/docs/<path:filename>')
def download_file(filename):
    return send_from_directory('docs', filename)

# 'Predict' button clicked
@app.route('/predict', methods=['POST'])
def predict():
    print('\nPredict button clicked')
    # Check if the request contains JSON data)
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    if 'stock_symbol' not in data:
        return jsonify({'error': 'No stock symbol provided'}), 400

    ticker = data['stock_symbol']

    try:
        model = models.get(ticker)
        if model is None:
            models[ticker] = Model(ticker)
            model = models[ticker]
        
        # Possible states are in_progress, pending, completed
        #   |   STATUS      |     FRONT END     |     BACK END      |
        #   | in_progress   |   Not affected    |  Updating         |
        #   | pending       |   Needs refresh   |  Update finished  |
        #   | completed     |   Refreshed       |  Update finished  |
        status = model.get_status() # Checks DB

        if status == 'in_progress':
            print('Model is currently being updated')
            # If the model is in progress, we return the last recommendation
            recommendation = 'Model is currently being updated, but here is the last recommendation:\n' + model.recommendation
        
        elif status == 'pending':
            print('Model has been updated, refreshing now...'),
            models.pop(ticker)
            models[ticker] = Model(ticker)
            model = models[ticker]
            model.update_completed()
            print('...done. Status set to ', model.get_status())

        elif status == 'completed':
            print('Model is up-to-date.')

        else: raise ValueError(f"Unknown status: {status}")
                
        return jsonify({
            'result': model.recommendation,
            'img1_path': f"{model.img1_path.replace('\\', '/')}?t={int(time.time())}",
            'img2_path': f"{model.img2_path.replace('\\', '/')}?t={int(time.time())}"
        })
    
    except ConnectionError as e:
        model = None
        return jsonify({'result': 'Connection error occurred, likely issue with yfinance.'})
    except Exception as e:
        model = None
        msg = 'An unknown error occurred: ' + str(e)
        return jsonify({'result': msg})

#   TODO This function can be removed after update overhaul is deployed
#--- Function: Ensure new column method is in place ---#
def check_columns():
    # Connect
    import sqlite3
    db_path = 'static/models/models.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if new column exists
    cursor.execute(f'PRAGMA table_info(models)')
    columns = [info[1] for info in cursor.fetchall()]
    if 'status' not in columns:
        print('New column must be added to models.db, updating now...')
        cursor.execute('''
            ALTER TABLE models ADD COLUMN status TEXT DEFAULT 'pending';
            ''')
        print('Table has been updated!')
    else:
        print('Table structure has been updated, consider removing old methods...')

#--------------------------------#

#--- First Boot ---#
if __name__ == '__main__':
    print('Starting Flask app...')
    # Create dirs
    img_dir = 'static/images'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    mdl_dir = 'static/models/'
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)
    
    # TODO this can be removed after update overhaul is deployed
    check_columns()

    app.run(debug=False)
#---------------------#