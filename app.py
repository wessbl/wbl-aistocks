from flask import Flask, render_template, request, jsonify, send_from_directory
from model.db_interface import DBInterface
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, 'static', 'images')

# Keep some logs B)
import logging
logging.basicConfig(filename='flask.log', level=logging.DEBUG)

# List of tickers
tickers = {}

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
        # Load the ticker information from the database
        dbi = DBInterface(os.path.join(BASE_DIR, 'static', 'models'))
        if ticker not in dbi.get_tickers():
            # TODO handle new ticker when front-end is ready
            return jsonify({'error': f'Ticker {ticker} not found in database. Please add it first.'}), 400
        model, result, last_update, status = dbi.load_model(ticker)
        print(f"Model loaded for {ticker}: result={result}, last_update={last_update}, status={status}")

        # Possible states are new, in_progress, pending, completed
        #   |   STATUS      |     FRONT END     |     BACK END      |
        #   | new           |   No Image Lookup |  Nothing          |
        #   | in_progress   |   Not affected    |  Updating         |
        #   | pending       |   Needs refresh   |  Update finished  |
        #   | completed     |   Refreshed       |  Update finished  |

        if status == 'new':
            recommendation = 'The AI will be trained on this ticker during the next update<br>(within 24 hours).'
            response = jsonify({
                'result': recommendation,
            })
            response.headers['Cache-Control'] = 'no-store'
            return response
        
        # Create text recommendation if it's stil a number
        if isinstance(result, float):
            recommendation = f"The AI recommends to <b>{'BUY' if result > 0 else 'SELL'}</b> {ticker}.<br>"
            recommendation += f"Predicted change over next 30 days: {result:.2f}%"
        else:
            recommendation = "Sorry, something went wrong and the recommendation came back empty."

        if status == 'in_progress':
            print('Model is currently being updated')
            # If the model is in progress, we return the last recommendation
            if result is None:
                recommendation = 'The AI is currently being trained on this ticker.<br>Please try again later.'
                response = jsonify({
                    'result': recommendation,
                })
                response.headers['Cache-Control'] = 'no-store'
                return response
            else:
                recommendation = '<i>Model is currently being updated, but here is the last recommendation:</i><br><br>' + recommendation
        
        # TODO pending status is vestigial but may come in handy for debugging
        elif status == 'pending':
            print('Model has been updated...', end=' ')
            dbi.set_status(ticker, 'completed')
            print('status set to completed.')

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
        return jsonify({'result': 'Connection error occurred, likely issue with yfinance.'})
    except Exception as e:
        msg = 'An unknown error occurred: ' + str(e)
        return jsonify({'result': msg})

# TODO front end not set up for this yet but the method may be useful
# @app.route('/add_ticker', methods=['POST'])
# def add_ticker():
#     stock_symbol = request.form.get('requested_stock_symbol', '').strip().upper()
#     if not stock_symbol:
#         return jsonify({'error': 'No stock symbol provided'}), 400

#     print(f"Adding ticker: {stock_symbol}")
#     try:
#         # Check if the model already exists
#         if stock_symbol in models:
#             return jsonify({'message': f'Model for {stock_symbol} already exists.'}), 200
        
#         # Create a new model and add it to the models dictionary
#         models[stock_symbol] = Model(stock_symbol, MODELS_PATH, IMG_PATH)
#         return jsonify({'message': f'Model for {stock_symbol} added successfully.'}), 200
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

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

        # TODO: Uncomment this block to run the updater immediately
        # else:
        #     print("app.py: Adding models and running model updater...")
        #     import model.updater # This will run the updater.py script
        #     print("app.py: Update completed successfully.")
    
    app.run(debug=True, use_reloader=False) # TODO set to False for production
#---------------------#