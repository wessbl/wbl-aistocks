import os
import sqlite3
import matplotlib
import os
from model.lstm_model import LSTMModel
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from db_interface import DBInterface

# A global collection of Models             TODO delete
class Models:
    models = {}
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _db_path = os.path.join(_base_dir, '../static/models/models.db')

    #--- Ctor ---#
    def populate(self):
        global models
        # If database isn't created yet, create FAANG stock models
        if not os.path.exists(self.db_path):
            tickers = ['AAPL', 'META', 'AMZN', 'NFLX', 'GOOGL']
            for ticker in tickers:
                self.get(ticker)
        
        # Create all the models for each db entry
        else:
            for ticker in db.get_tickers():
                self.get(ticker)
    #---------------------------#
    
    #--- Get Model ---#
    def get(self, ticker):
        global models
        model = models.get(ticker)

        # Create new model if needed
        if model is None:
            models[ticker] = Model(ticker)
            model = models.get(ticker)
        
        return model
    #----------------#

    #--- Update ---#
    def update(self):
        global models
        for model in models:
            model.update()

#---------------------------------------------#


# A wrapper class for LSTMModels that generates images
class Model:
    # LSTM Vars
    ticker = None
    recommendation = None
    status = None
    _lstm = None
    _prediction = None
    _mirror = None
    _db = DBInterface()

    # Paths
    img1_path = None
    img2_path = None
    _lstm_path = None
    
    #--- Constructor ---#
    def __init__(self, ticker):
        self.ticker = ticker

        # Create paths
        self._lstm_path = 'static/models/' + ticker + '.keras'
        self._lstm_path = 'static/models/' + ticker + '.keras'
        self.img1_path = 'static/images/' + ticker + 'pred.png'
        self.img2_path = 'static/images/' + ticker + 'mirr.png'

        # First, try to load an existing model
        try:
            model, last_update, self.recommendation, status = db.load(ticker)
            self._model = LSTMModel(ticker, model, last_update, status)

            # Attempt update, generate new outputs if needed
            updated, text = self._model.update_model()
            print("\nModel Updated:\t", updated, "\nExplanation:\t", text)
            if updated:
                self.generate_output(self._model)

        except Exception as e:
            print(str(e))
            print("Could not find ticker in database:\t", ticker)
            print("Creating new model...")
            # Train a new model on 10+ years of data
            self._lstm = LSTMModel(ticker)
            self.generate_output(self._lstm)
        self._save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
    #-------------------------------#
    
    #--- Function: Predict, generate imgs, save ---#
    def generate_output(self, model):
        # Make prediction (data) & recommendation (text)
        prediction = model.make_prediction()
        self.recommendation = model.buy_or_sell(prediction)
        
        mirror = model.mirror_data()
        self._generate_prediction(model, prediction)
        self._generate_mirror(model, mirror)
    #----------------------------------------------#

    #--- Function: Create prediction image ---#
    def _generate_prediction(self, model, prediction):
        # Get variables
        zoom_data = model.orig_data[-model.time_step:]
        dividing_line = model.time_step - 1
        end = dividing_line + len(prediction)
        plt.figure(figsize=(6, 3))
        plt.title(f'Prediction - {model.ticker}')
        plt.axvline(x=dividing_line, color='grey', linestyle=':', label=model.last_update.strftime("%m/%d/%Y"))
        plt.plot(zoom_data, label="Actual Price")
        plt.plot(np.arange(dividing_line, end), prediction, label='Prediction')
        plt.legend()
        
        # Save image
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
        self._lstm.train(epochs, mse_threshold=threshold)
        self.generate_output(self._lstm)
        self._save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
    #----------------------------------------------#