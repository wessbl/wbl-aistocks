from flask import Flask, render_template, request, jsonify
from model.model import Model
import os

# Suppresses INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Dictionary of models ticker : Model
models = {}

app = Flask(__name__)

# Show /index.html
@app.route('/')
def home():
    return render_template('index.html')

# 'Predict' button clicked
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'stock_symbol' not in data:
        return jsonify({'error': 'No stock symbol provided'}), 400

    stock_symbol = data['stock_symbol']

    try:
        model = Model(stock_symbol)
        # TODO do we need this?
        # img1_path = os.path.join(img_dir, f'{ticker}_complete.png')
        # img2_path = os.path.join(img_dir, f'{ticker}_prediction.png')
        
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
    tickers = ['AAPL', 'META', 'AMZN', 'NFLX', 'GOOGL']
    for ticker in tickers:
        models[ticker] = Model(ticker)
    # model = Model('AAPL')
    # model = LSTMModel('META', mdl_dir)
    # model = LSTMModel('AMZN', mdl_dir)
    # model = LSTMModel('NFLX', mdl_dir)
    # model = LSTMModel('GOOGL', mdl_dir)
    return
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
    # os.remove('static/models/models.db')      # TODO delete line
    db_path = 'static/models/models.db'
    if not os.path.exists(db_path):
        train_models()
    app.run(debug=True)
#---------------------#