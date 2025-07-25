from flask import Flask, render_template, request, jsonify, send_from_directory
from model.model import Model
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'static', 'models')
IMG_PATH = os.path.join(BASE_DIR, 'static', 'images')

# Keep some logs B)
import logging
logging.basicConfig(filename='flask.log', level=logging.DEBUG)

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
    # Check if the request contains JSON data)
    data = request.json
    if not data:
        return jsonify({'error': 'Predict button clicked but no data provided'}), 400
    if 'stock_symbol' not in data:
        return jsonify({'error': 'Predict button clicked but no stock symbol provided'}), 400

    ticker = data['stock_symbol']
    print(f"\nPredict button clicked for ticker: {ticker}")

    try:
        model = models.get(ticker)
        if model is None:
            models[ticker] = Model(ticker, MODELS_PATH, IMG_PATH) # TODO create a shallow copy that doesn't read/write to keras file
            model = models[ticker]
        
        # Possible states are new, in_progress, pending, completed
        #   |   STATUS      |     FRONT END     |     BACK END      |
        #   | new           |   No Image Lookup |  Nothing          |
        #   | in_progress   |   Not affected    |  Updating         |
        #   | pending       |   Needs refresh   |  Update finished  |
        #   | completed     |   Refreshed       |  Update finished  |
        status = model.get_status() # Checks DB
        recommendation = model.recommendation

        if status == 'new':
            recommendation = 'Model will be trained on the next update (within 24 hours).'

        elif status == 'in_progress':
            print('Model is currently being updated')
            # If the model is in progress, we return the last recommendation
            recommendation = '<i>Model is currently being updated, but here is the last recommendation:</i><br><br>' + model.recommendation
        
        elif status == 'pending':
            print('Model has been updated, refreshing now...'),
            models.pop(ticker)
            models[ticker] = Model(ticker, MODELS_PATH, IMG_PATH)
            model = models[ticker]
            model.update_completed()
            recommendation = model.recommendation
            print('...done. Status set to ', model.get_status())

        elif status == 'completed':
            print('Model is up-to-date.')

        else: raise ValueError(f"Unknown status: {status}")

        # Prepare image paths
        img1_path = 'static/images/' + ticker + 'pred.png'
        img1_path = img1_path.replace('\\', '/')
        img2_path = 'static/images/' + ticker + 'mirr.png'
        img2_path = img2_path.replace('\\', '/')

        # Return the recommendation and image paths
        response = jsonify({
            'result': recommendation,
            'img1_path': f"{img1_path}?t={int(time.time())}",
            'img2_path': f"{img2_path}?t={int(time.time())}"
        })
        response.headers['Cache-Control'] = 'no-store'
        return response
    
    except ConnectionError as e:
        model = None
        return jsonify({'result': 'Connection error occurred, likely issue with yfinance.'})
    except Exception as e:
        model = None
        msg = 'An unknown error occurred: ' + str(e)
        return jsonify({'result': msg})

@app.route('/add_ticker', methods=['POST'])
def add_ticker():
    stock_symbol = request.form.get('requested_stock_symbol', '').strip().upper()
    if not stock_symbol:
        return jsonify({'error': 'No stock symbol provided'}), 400

    print(f"Adding ticker: {stock_symbol}")
    try:
        # Check if the model already exists
        if stock_symbol in models:
            return jsonify({'message': f'Model for {stock_symbol} already exists.'}), 200
        
        # Create a new model and add it to the models dictionary
        models[stock_symbol] = Model(stock_symbol, MODELS_PATH, IMG_PATH)
        return jsonify({'message': f'Model for {stock_symbol} added successfully.'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


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
    
    # Check for version updates
    print("Checking for version updates...")
    if os.path.exists('fs_version_update.py'):
        import fs_version_update
        result = fs_version_update.update_fs()
        if not result:
            print("Update failed, exiting app.")
            exit(1)
        else:
            print("app.py: Adding models and running model updater...")
            models['AAPL'] = Model('AAPL', MODELS_PATH, IMG_PATH)
            models['GOOGL'] = Model('GOOGL', MODELS_PATH, IMG_PATH)
            models['META'] = Model('META', MODELS_PATH, IMG_PATH)
            models['AMZN'] = Model('AMZN', MODELS_PATH, IMG_PATH)
            models['NFLX'] = Model('NFLX', MODELS_PATH, IMG_PATH)
            # TODO: Uncomment the next line to run the updater immediately
            # import model.updater # This will run the updater.py script
            print("app.py: Update completed successfully.")
    
    app.run(debug=False)
#---------------------#