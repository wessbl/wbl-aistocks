import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model.lstm_model import LSTMModel


# A wrapper class for LSTMModels that generates images
class Model:
    # LSTM Vars
    ticker = None
    recommendation = None
    _model = None
    _prediction = None
    _mirror = None

    # Paths
    img1_path = None
    img2_path = None
    _db_path = 'static/models/models.db'
    _model_path = None
    
    #--- Constructor ---#
    def __init__(self, ticker):
        # Create paths
        self._model_path = 'static/models/' + ticker + '.keras'
        self.img1_path = 'static/images/' + ticker + 'pred.png'
        self.img2_path = 'static/images/' + ticker + 'mirr.png'

        # First, try to load an existing model
        try:
            model, last_update, self.recommendation = self.load(ticker)
            self._model = LSTMModel(ticker, model)

            # Attempt update
            updated, text = self._model.update_model()
            print("\nModel Updated:\t", updated, "\nExplanation:\t", text)
            if updated:
                self.generate_output(self._model)

        except:
            # Train a new model on 10+ years of data
            self._model = LSTMModel(ticker)
        self.save(self.ticker, self.model, self.last_update, self.result)
    #-------------------------------#
    
    #--- Function: Predict, generate imgs, save ---#
    def generate_output(self, model):
        # Make prediction (data) & recommendation (text)
        prediction = model.make_prediction(self.prediction_len)
        self.recommendation = model.buy_or_sell(prediction)
        
        mirror = model.mirror_data(model)
        self._generate_prediction(model)
        self._generate_mirror(model)
    #----------------------------------------------#

    #--- Function: Create prediction image ---#
    def _generate_prediction(self, model):
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
        plt.savefig(self.img1_path)
        plt.close()
    #------------------------------------------#

    #--- Function: Create price history image ---#
    def _generate_mirror(self, model):
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
        plt.savefig(self.img2_path)
        plt.close()
    #-----------------------------------------------#

    #--- Function: Save to DB ---#
    def _save(self, ticker, model, last_update, result=''):
        # Save model as file
        model.save(self.model_filename)

        # Read the model file as binary
        with open(self.model_filename, 'rb') as f:
            model_binary = f.read()

        # Get text version of last_update
        last_update_txt = last_update.strftime("%Y-%m-%d")

        # Database connection
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Handle blob -> model
        cursor.execute('''
            SELECT model FROM models WHERE ticker = ?''',
            (ticker,))
        data = cursor.fetchone()[0]
        with open(self.model_filename, 'wb') as file:
            file.write(data)
        model = load_model(self.model_filename)

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
            if self.verbose: print("\nLoaded data from database!\nTicker:\t\t", ticker,
                "\nModel:\t\t", model, "\nLast Update:\t", last_update,
                "\nResult:\t\t", result)
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
