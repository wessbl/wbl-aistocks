import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model.lstm_model import LSTMModel
from model.db_interface import DBInterface

# A wrapper class for LSTMModels that generates images
class Model:
    # LSTM Vars
    ticker = None
    recommendation = None
    _status = None
    _lstm = None
    _prediction = None
    _mirror = None
    _db = None

    # Paths
    img1_path = None
    img2_path = None
    _lstm_path = None
    
    #--- Constructor ---#
    def __init__(self, ticker, MODELS_PATH, IMG_PATH):
        self.ticker = ticker
        self._db = DBInterface(MODELS_PATH)

        # Create paths
        self._lstm_path = os.path.join(MODELS_PATH, ticker, '.keras')
        self.img1_path = os.path.join(IMG_PATH, (ticker + 'pred.png'))
        self.img2_path = os.path.join(IMG_PATH, (ticker + 'mirr.png'))

        # First, try to load an existing model
        try:
            keras_model, self.recommendation, last_update, self._status = self._db.load_model(ticker)
            # if self._status == 'new':
            #     self._lstm = LSTMModel(ticker, keras_model, last_update, self._status)
            # else: 
            self._lstm = LSTMModel(ticker, keras_model, last_update, self._status)

        except Exception as e:
            # If the model doesn't exist, create a new one
            print(f"Error loading model for {ticker}: {e}")
            print(f"Could not find {ticker} in database. Creating new model...", end=' ')
            self._lstm = LSTMModel(ticker)
            self._db.save_model(self.ticker, self._lstm, status='new')
            print("done.")
    #-------------------------------#
    
    #--- Function: Predict, generate imgs, save ---#
    def generate_output(self):
        # Make prediction (data) & recommendation (text)
        print(f"Generating output for {self.ticker}...")
        prediction = self._lstm.make_prediction()
        percent = self._lstm.percentage_change(prediction)
        buy = percent > 0
        percent = str(f"{percent:.2f}")
        if buy:
            self.recommendation = "<b>Buy</b><br>AIStockHelper says this stock will change by " + percent + "%."
        else:
            self.recommendation = "<b>Sell</b><br>AIStockHelper says this stock will change by " + percent + "%."

        # Save the prediction to the database
        today = self._db.today_id()
        if today == -1:
            print("Error: Could not get today's day ID from the database.")
            return
        for i in range(len(prediction)):
            self._db.save_prediction(self.ticker, today, today+i, prediction[i], buy)
        
        # Save buy/sell recommendation to the database
        # TODO save 'buy' variable
        
        # Create images
        mirror = self._lstm.mirror_data()
        self._generate_prediction(self._lstm, prediction)
        self._generate_mirror(self._lstm, mirror)
    #----------------------------------------------#

    #--- Function: Create prediction image ---#
    def _generate_prediction(self, _lstm, prediction):
        # Get variables
        zoom_data = _lstm.orig_data[-_lstm.time_step:]
        dividing_line = _lstm.time_step - 1
        end = dividing_line + len(prediction)
        date = self._db.get_day_string(self._db.today_id())
        plt.figure(figsize=(6, 3))
        plt.title(f'Prediction - {_lstm.ticker}')
        plt.axvline(x=dividing_line, color='grey', linestyle=':', label=date)
        plt.plot(zoom_data, label="Actual Price")
        plt.plot(np.arange(dividing_line, end), prediction, label='Prediction')
        plt.legend()
        
        # Save image
        # TODO remove, shouldn't be needed
        if not os.path.exists(os.path.dirname(self.img1_path)):
            print("Directory doesn't exist??!")
            raise FileNotFoundError(f"Directory for {self.img1_path} does not exist.")
        plt.savefig(self.img1_path)
        plt.close()
    #------------------------------------------#

    #--- Function: Create price history image ---#
    def _generate_mirror(self, model, mirror):
        # Start + end dates for 'mirror' display
        start = model.time_step
        end = start + len(mirror)

        # Create matplot
        plt.figure(figsize=(6, 3))
        plt.title(f'Model Against Actual Price - {model.ticker}')
        plt.plot(model.orig_data, label="Actual Price")
        plt.plot(np.arange(start, end), mirror, label='Model Prediction')
        plt.legend()

        # Save as file
        plt.savefig(self.img2_path)
        plt.close()
    #-----------------------------------------------#

    #--- Function: Train model further ---#
    def train(self, epochs, threshold=0):
        # Check if the model needs to be updated
        if not self._lstm.needs_update():
            print(f"Model for {self.ticker} is up-to-date, no training needed.")
            return
        # Set status to in_progress
        self._set_status(1)
        # Train, generate output, and save to DB
        self._lstm.train(epochs, mse_threshold=threshold)

        # TODO 0.8 - Calculate accuracy here

        self.generate_output()
        self._db.save_model(self.ticker, self._lstm, self._lstm.last_update, self.recommendation, 'pending')
        self._status = 'pending'
    #----------------------------------------------#

    #--- Function: Change status to completed ---#
    def update_completed(self):
        self._set_status(3)
    #----------------------------------------------#

    #--- Function: Change status ---#
    def _set_status(self, status_int):
        temp_status = ''
        if status_int == 1:
            temp_status = 'in_progress'
        elif status_int == 2:
            temp_status = 'pending'
        elif status_int == 3:
            temp_status = 'completed'
        else:
            raise ValueError("Invalid status integer.")
        
        self._db.set_status(self.ticker, temp_status)
        self._status = temp_status
    #----------------------------#

    #--- Function: Check status ---#
    def get_status(self):
        return self._db.get_status(self.ticker)
    #----------------------------#