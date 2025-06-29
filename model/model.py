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
            keras_model, self.recommendation, last_update, self._status = self._db.load(ticker)
            self._lstm = LSTMModel(ticker, keras_model, last_update, self._status)

        except Exception as e:
            print(e)
            print("Error:\t", str(e))
            print("Could not find ticker in database:\t", ticker)
            print("Creating new model...")
            # Train a new model on 10+ years of data
            self._lstm = LSTMModel(ticker)
            self._db.save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
    #-------------------------------#
    
    #--- Function: Predict, generate imgs, save ---#
    def generate_output(self):
        # Make prediction (data) & recommendation (text)
        print(f"Generating output for {self.ticker}...")
        prediction = self._lstm.make_prediction()
        self.recommendation = self._lstm.buy_or_sell(prediction)
        
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
        plt.figure(figsize=(6, 3)) # TODO check new size
        plt.title(f'Prediction - {_lstm.ticker}')
        plt.axvline(x=dividing_line, color='grey', linestyle=':', label=_lstm.last_update.strftime("%m/%d/%Y"))
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
        plt.figure(figsize=(6, 3)) # TODO check new size
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
        self._set_status(1)
        # Train the LSTM model
        self._lstm.train(epochs, mse_threshold=threshold)
        self.generate_output()
        self._db.save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
        self._set_status(2)
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