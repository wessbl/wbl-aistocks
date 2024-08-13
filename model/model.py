import matplotlib
from model.lstm_model import LSTMModel
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from db_interface import DBInterface

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
            keras_model, self.recommendation, last_update, self.status = self._db.load(ticker)
            self._lstm = LSTMModel(ticker, keras_model, last_update)

        except Exception as e:
            print(str(e))
            print("Could not find ticker in database:\t", ticker)
            print("Creating new model...")
            # Train a new model on 10+ years of data
            self._lstm = LSTMModel(ticker)
        self.generate_output()
        self._db.save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
    #-------------------------------#
    
    #--- Function: Predict, generate imgs, save ---#
    def generate_output(self):
        global _lstm
        # Make prediction (data) & recommendation (text)
        prediction = _lstm.make_prediction()
        self.recommendation = _lstm.buy_or_sell(prediction)
        
        mirror = _lstm.mirror_data()
        self._generate_prediction(_lstm, prediction)
        self._generate_mirror(_lstm, mirror)
    #----------------------------------------------#

    #--- Function: Create prediction image ---#
    def _generate_prediction(self, _lstm, prediction):
        # Get variables
        zoom_data = _lstm.orig_data[-_lstm.time_step:]
        dividing_line = _lstm.time_step - 1
        end = dividing_line + len(prediction)
        plt.figure(figsize=(12, 6)) # TODO check new size
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
        plt.figure(figsize=(12, 6)) # TODO check new size
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
        self.generate_output()
        self._db.save(self.ticker, self._lstm, self._lstm.last_update, self.recommendation)
    #----------------------------------------------#

    #--- Function: Change status to completed ---#
    def update_completed(self):
        self._set_status(3)
    #----------------------------------------------#

    #--- Function: Change status ---#
    def _set_status(self, status_int):
        status = ''
        if status_int == 1:
            status = 'in_progress'
        elif status_int == 2:
            status = 'pending'
        elif status_int == 3:
            status = 'completed'
        
        self._db.set_status(self.ticker, status)
    #----------------------------#

    #--- Function: Perform a daily update ---#
    def update(self):
        self._set_status(1)
        self.train(50, .0002)
        # Status is set to pending automatically

        self.generate_output()
    #----------------------------------------#