from flask import Flask, render_template, request, jsonify
from model.lstm_model import LSTMModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations

app = Flask(__name__)

img_dir = 'static/images'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
mdl_dir = 'static/models/'
if not os.path.exists(mdl_dir):
    os.makedirs(mdl_dir)

@app.route('/')
def home():
    return "hello world!"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'stock_symbol' not in data:
        return jsonify({'error': 'No stock symbol provided'}), 400

    stock_symbol = data['stock_symbol']

    try:
        model = LSTMModel(stock_symbol, mdl_dir)
        result = model.result
        ticker = model.ticker
        img1_path = os.path.join(img_dir, f'{ticker}_complete.png')
        img2_path = os.path.join(img_dir, f'{ticker}_prediction.png')
        show_mirror(model, img1_path)
        show_prediction(model, img2_path)
        
        return jsonify({
            'result': str(result),
            'img1_path': img1_path.replace('\\', '/'),
            'img2_path': img2_path.replace('\\', '/')
        })
    except ConnectionError as e:
        model = None
        return jsonify({'result': 'Connection error occurred, likely issue with yfinance.'})
    except Exception as e:
        model = None
        msg = 'An unknown error occurred: ' + e
        return jsonify({'result': msg})


#--- Function: Create prediction image ---#
def show_prediction(model, img_path):
    # Get variables
    orig_data = model.orig_data
    time_step = model.time_step
    ticker = model.ticker
    prediction = model.last_pred

    zoom_data = orig_data[-time_step:]
    dividing_line = time_step - 1
    end = dividing_line + len(prediction)
    plt.figure(figsize=(6, 3))
    plt.title(f'Prediction - {ticker}')
    plt.axvline(x=dividing_line, color='grey', linestyle=':', label='Last Close')
    plt.plot(zoom_data, label="Actual Price")
    plt.plot(np.arange(dividing_line, end), prediction, label='Prediction')
    plt.legend()
    
    # Save image
    plt.savefig(img_path)
    plt.close()
#------------------------------------------#


#--- Function: Create price history image ---#
def show_mirror(model, img_path):
    # Get variables
    ticker = model.ticker
    time_step = model.time_step
    orig_data = model.orig_data
    mirror = model.last_mirror

    # Start + end dates for 'mirror' display
    start = time_step
    end = start + len(mirror)

    # Create matplot
    plt.figure(figsize=(6, 3))
    plt.title(f'Model Against Actual Price - {ticker}')
    plt.plot(orig_data, label="Actual Price")
    plt.plot(np.arange(start, end), mirror, label='Model Prediction')
    plt.legend()

    # Save as file
    plt.savefig(img_path)
    plt.close()
#-----------------------------------------------#

def train_models():
    model = LSTMModel('AAPL', mdl_dir)
    model = LSTMModel('META', mdl_dir)
    model = LSTMModel('AMZN', mdl_dir)
    model = LSTMModel('NFLX', mdl_dir)
    model = LSTMModel('GOOGL', mdl_dir)
    return

if __name__ == '__main__':
    db_path = 'static/models/models.db'
    if not os.path.exists(db_path):
        train_models()
    app.run(debug=True)