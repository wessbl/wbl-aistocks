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

#--- First Boot ---#
if __name__ == '__main__':
    # Create dirs
    img_dir = 'static/images'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    mdl_dir = 'static/models/'
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    Models.populate()
    app.run(debug=False)
#---------------------#

