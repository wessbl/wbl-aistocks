import sqlite3
import matplotlib
import os
from model.lstm_model import LSTMModel
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model


# A wrapper class for LSTMModels that generates images
class Model:
    # LSTM Vars
    ticker = None
    recommendation = None
    _lstm = None
    _prediction = None
    _mirror = None

    # Paths
    img1_path = None
    img2_path = None
    _db_path = 'static/models/models.db'
    _lstm_path = None
    
    #--- Constructor ---#
    def __init__(self, ticker):
        self.ticker = ticker

        # Create paths
        self._lstm_path = 'static/models/' + ticker + '.keras'
        self.img1_path = 'static/images/' + ticker + 'pred.png'
        self.img2_path = 'static/images/' + ticker + 'mirr.png'

        # First, try to load an existing model
        try:
            model, last_update, self.recommendation = self._load(ticker)
            self._lstm = LSTMModel(ticker, model, last_update)

            # Attempt update, generate new outputs if needed
            updated, text = self._lstm.update_model()
            print("\nModel Updated:\t", updated, "\nExplanation:\t", text)
            if updated:
                self.generate_output(self._lstm)

        except Exception as e:
            print(str(e))
            print("Could not find ticker in database:\t", ticker)
            print("Creating new model...")
            # Train a new model on 10+ years of data
            self._lstm = LSTMModel(ticker)
            self.generate_output(self._lstm)
        self._save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
    #-------------------------------#
    
    #--- Function: Train model further ---#
    def train(self, epochs, threshold=0):
        self._lstm.train(epochs, mse_threshold=threshold)
        self.generate_output(self._lstm)
        self._save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
    #----------------------------------------------#

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

    #--- Function: Save to DB ---#
    def _save(self, ticker, model, last_update, result=''):
        # Save model as file
        model._model.save(self._lstm_path)

        # Read the model file as binary
        with open(self._lstm_path, 'rb') as f:
            model_binary = f.read()

        # Get text version of last_update
        last_update_txt = last_update.strftime("%Y-%m-%d")

        # Database connection
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
            ticker TEXT PRIMARY KEY,
            model BLOB,
            last_update TEXT,
            result TEXT
            )
        ''')

        # Store the model in the database
        cursor.execute('''
            INSERT OR REPLACE INTO models (ticker, model, last_update, result)
            VALUES (?, ?, ?, ?)''',
            (ticker, model_binary, last_update_txt, result))
        conn.commit()
        conn.close()
    #------------------------------#

    #--- Function: Load from DB ---#
    def _load(self, ticker):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Handle blob -> model
        cursor.execute('''
            SELECT model FROM models WHERE ticker = ?''',
            (ticker,))
        data = cursor.fetchone()[0]
        with open(self._lstm_path, 'wb') as file:
            file.write(data)
        model = load_model(self._lstm_path)

        # Get update & result
        cursor.execute('''
            SELECT last_update, result FROM models WHERE ticker = ?''',
            (ticker,))
        row = cursor.fetchone()
        conn.close()

        if row:
            last_update_text, result = row

            # Last update txt -> Timestamp
            last_update = pd.Timestamp(last_update_text)

            # Done
            print("\nModel for "+ticker+" loaded from database...")
            # print("\nLoaded data from database!\nTicker:\t\t", ticker,
            #     "\nModel:\t\t", model, "\nLast Update:\t", last_update,
            #     "\nResult:\t\t", result)
            return model, last_update, result

        else:
            raise ValueError("Model could not be found in the database.")
    #------------------------------#

    #--- Function: drop SQL tables ---#
    def drop_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS models')
        conn.commit()
        conn.close()
    #------------------------------#

