import os
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
    _status = None
    _lstm = None
    _prediction = None
    _mirror = None
    _db = None
    _yf = None

    # Paths
    img1_path = None
    img2_path = None
    
    #--- Constructor ---#
    def __init__(self, ticker, db, yf, IMG_PATH):
        self.ticker = ticker
        self._db = db
        self._yf = yf

        # Create paths
        self.img1_path = os.path.join(IMG_PATH, (ticker + 'pred.png'))
        self.img2_path = os.path.join(IMG_PATH, (ticker + 'mirr.png'))

        # First, try to load an existing model
        try:
            keras_model, self.recommendation, last_update, self._status = self._db.load_model(ticker)
            # if self._status == 'new':
            #     self._lstm = LSTMModel(ticker, keras_model, last_update, self._status)
            # else: 
            self._lstm = LSTMModel(ticker, keras_model, last_update, self._status, self._yf)

        # If the model doesn't exist, create a new one
        except Exception as e:
            print("Creating new model...", end=' ')
            self._lstm = LSTMModel(ticker, yf=self._yf)
            self._db.save_model(self.ticker, self._lstm, status='new') # TODO this will need to be moved to app.py
            print("done.")
    #-------------------------------#

    # TODO is this needed?
    #--- Function: Prepare for new day ---#
    def save_actual_price(self, for_day, price):
        # Get the date version of day
        day_str = self._db.get_day_string(for_day)
        # Get the price
        price = self._yf.get_price(self.ticker, day_str)
        if price is None:
            raise ValueError(f"Could not retrieve price for {self.ticker} on day {for_day}.")

        self._db.save_actual_price(self.ticker, for_day, price)
    #-------------------------------------#
    
    #--- Function: Predict, generate imgs, save ---#
    def generate_output(self, day):
        # Make prediction (data) & recommendation (text)
        print(f"Generating output for {self.ticker}...")
        prediction = self._lstm.make_prediction()
        # TODO probably most of this is no longer needed
        percent = self._lstm.percentage_change(prediction)
        self.recommendation = percent
        buy = True if percent > 0 else False

        for i in range(1, len(prediction)): # Skip the first prediction (current price)
            # TODO remove after testing
            # print(f"day:{day} i:{i} prediction:{prediction[i]} buy:{buy}")
            # # sleep for 1 second to avoid overwhelming the database
            # import time
            # time.sleep(1)
            self._db.save_prediction(self.ticker, day, day+i, float(prediction[i]), bool(buy))
        
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
        date = self._db.get_day_string(self._db.today_num())
        plt.figure(figsize=(6, 3))
        plt.title(f'Prediction - {_lstm.ticker}')
        plt.axvline(x=dividing_line, color='grey', linestyle=':', label=date)
        plt.plot(zoom_data, label="Actual Price")
        plt.plot(np.arange(dividing_line, end), prediction, label='Prediction')
        plt.legend()
        
        # Save image
        if not os.path.exists(os.path.dirname(self.img1_path)):
            print("Directory doesn't exist!")
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
    def train(self, epochs=5, threshold=0):
        # Train model starting with first missing date in prediction table
        # TODO 0.8 check for model's first date instead of first date in DB
        print(f"Model: Epochs={epochs}, Threshold={threshold}") # TODO BUG it's doing 5 epochs when being told to do 1 from updater
        dates = self._db.all_dates()
        days = self._db.all_days()
        first_missing_day = self._db.train_start_day(self.ticker)
        if first_missing_day == -1:
            print(f"Model: {self.ticker} is up-to-date, no training needed.")
            return
        print(f"First missing day for {self.ticker} is {first_missing_day}.") #TODO may be irrelevant
        
        # Train on all days from first_missing_day to the end
        start_index = days.index(first_missing_day)
        for i in range(start_index, len(days)): # BUG first_missing_day is being used as index
            print(f"Training {self.ticker} on day {days[i]}: {dates[i]}...")
            self._lstm.train(epochs, dates[i], threshold) # TODO double-check what epochs is set to
            self.generate_output(days[i])
            self._db.save_actual_price(self.ticker, days[i], self._yf.get_price(self.ticker, dates[i]))

        # TODO remove after testing
        # if status is 'new', train the model up to 2025-10-30 then have it predict every day after
        # if self._lstm.status == 'new':
        #     # LSTM needs date, DB needs day int. Get both
        #     dates = self._db.all_dates()
        #     days = self._db.all_days()
        #     self._lstm.train(epochs, dates[0], threshold)
        #     for i in range(len(days)):
        #         print(f"Training model for {self.ticker} on {dates[i]}...")
        #         self._lstm.train(1, dates[i])
        #         self.generate_output(days[i])
        #         self._db.save_actual_price(self.ticker, days[i], self._yf.get_price(self.ticker, dates[i]))
            
        # else:
        #     # Check if the model needs to be updated
        #     if not self._lstm.needs_update():
        #         print(f"Model for {self.ticker} is up-to-date, no training needed.")
        #         return
        #     # Set status to in_progress
        #     self._set_status(1)
        #     # Train, generate output, and save to DB
        #     self._lstm.train(epochs, mse_threshold=threshold)
        #     self.generate_output(self._db.today_num())

        # Save model, which is still in progress
        self._db.save_model(self.ticker, self._lstm, self._lstm.last_update, self.recommendation, 'in_progress')
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