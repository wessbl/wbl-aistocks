from flask import Flask, render_template, request, jsonify, send_from_directory
from model.model import Model
import os

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
    global models
    data = request.json
    if 'stock_symbol' not in data:
        return jsonify({'error': 'No stock symbol provided'}), 400

    stock_symbol = data['stock_symbol']

    try:
        model = models.get(stock_symbol)
        if model is None:
            models[stock_symbol] = Model(stock_symbol)
            model = models.get(stock_symbol)
        
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

#--- Function: Train FAANG models ---#
def train_models():
    global models
    tickers = ['AAPL', 'META', 'AMZN', 'NFLX', 'GOOGL']
    for ticker in tickers:
        models[ticker] = Model(ticker)
#------------------------------------#

#--- First Boot ---#
if __name__ == '__main__':
    # Create dirs
    img_dir = 'static/images'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    mdl_dir = 'static/models/'
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    # Train 5 models
    db_path = 'static/models/models.db'
    if not os.path.exists(db_path):
        train_models()
    app.run(debug=False)
#---------------------#

