from flask import Flask, render_template, request, jsonify, send_from_directory
from model.model import Model
import os
import time

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
            models[ticker] = Model(ticker)
            model = models[ticker]
        
        # Possible states are in_progress, pending, completed
        #   |   STATUS      |     FRONT END     |     BACK END      |
        #   | in_progress   |   Not affected    |  Updating         |
        #   | pending       |   Needs refresh   |  Update finished  |
        #   | completed     |   Refreshed       |  Update finished  |
        status = model.get_status() # Checks DB
        recommendation = model.recommendation

        if status == 'in_progress':
            print('Model is currently being updated')
            # If the model is in progress, we return the last recommendation
            recommendation = '<i>Model is currently being updated, but here is the last recommendation:</i><br><br>' + model.recommendation
        
        elif status == 'pending':
            print('Model has been updated, refreshing now...'),
            print(f"\t[Pre-pop] Model ID: {id(models.get(ticker))}")
            models.pop(ticker, None)
            print(f"\t[Post-pop] Exists? {ticker in models}")
            new_model = Model(ticker)
            new_model.update_completed()
            models[ticker] = new_model
            recommendation = model.recommendation
            print('...done. Status set to ', model.get_status())

        elif status == 'completed':
            print(f'Model is up-to-date. There are {models.__len__()} models in memory.')

        else: raise ValueError(f"Unknown status: {status}")

        # Prepare image paths
        img1_path = model.img1_path.replace('\\', '/')
        img2_path = model.img2_path.replace('\\', '/')

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
    
    app.run(debug=False)
#---------------------#