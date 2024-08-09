from flask import Flask, render_template, request, jsonify, send_from_directory
from model.model import Models
import os

# Suppresses INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    data = request.json
    if 'stock_symbol' not in data:
        return jsonify({'error': 'No stock symbol provided'}), 400

    stock_symbol = data['stock_symbol']

    try:
        model = Models.get(stock_symbol)

        # Possible states are in_progress, pending, completed
        #   in_progress:    Models are currently updating, front-end not affected
        #   pending:        Updates are finished, front-end needs refresh
        #   completed:      Updates are finished and front-end refreshed
        if model.status == 'pending':
            print('Model has been updated, refreshing now...')
            model = None
            # TODO get info from model...
        
        return jsonify({
            'result': model.recommendation,
            'img1_path': model.img1_path.replace('\\', '/'),
            'img2_path': model.img2_path.replace('\\', '/')
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
    # Create dirs
    img_dir = 'static/images'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    mdl_dir = 'static/models/'
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)
    
    # TODO this can be removed after update overhaul is deployed
    check_columns()

    Models.populate()
    app.run(debug=False)
#---------------------#

