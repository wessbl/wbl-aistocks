import yfinance as yf
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
import logging
import os
logging.getLogger('tensorflow').setLevel(logging.ERROR) # Set tf logs to error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Turn off oneDNN custom operations
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    # LSTM Performance Variables
    ticker = None               # MSFT | DAC | AAPL
    time_step = 50              # 50  | 100 |
    _epochs = 10                # 10  | 50  | 100   #TODO switch to 100 after server switch
    _update_epoch = 1           # how many epochs for an update
    _prediction_len = 5         # how many days to predict
    _start_date = '2015-01-01'
    last_update = None
    _model = None
    orig_data = None
    _scaled_data = None
    prediction = None
    recommendation = None

    X = np.array([])
    scaler = MinMaxScaler(feature_range=(0,1))

    #--- Constructor ---#
    def __init__(self, ticker, model=None, last_update=None, status=None):
        # Check for valid model & attempt update
        if model is not None:
            self.ticker = ticker
            self._model = model
            self.last_update = last_update
            self.status = status
            print("Model loaded.")
        else:
            # Get data & train brand-new model
            self.ticker = ticker
            self.preprocess()
            print("Training new model...")
            self._model = self.train_model(model)
    #------------------------------#

    #--- Function: Create the dataset for LSTM ---#
    def create_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.time_step - 1):
            X.append(data[i:(i + self.time_step), 0])
            y.append(data[i + self.time_step, 0])
        return np.array(X), np.array(y)
    #---------------------------------------------#

    #--- Function: Preprocess the latest data ---#
    def preprocess(self):
        # Get all data from start through last close (yf excludes end date)
        today = pd.Timestamp.now().date()
        if today >= self.last_close():
            df = yf.download(self.ticker, start=self._start_date)
        else: 
            df = yf.download(self.ticker, start=self._start_date, end=today)
        if (len(df) == 0): raise ValueError("Cannot download from yfinance")

        # Create global orig_data, scaler, scaled_data, X, y
        orig_data = df['Close'].values
        self.orig_data = orig_data.reshape(-1, 1)    # Reshape into a 2d array: [[1], [2], [3]]
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self._scaled_data = self.scaler.fit_transform(self.orig_data)

        # Create Datasets to feed LSTM
        self.X, self.y = self.create_dataset(self._scaled_data)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
    #---------------------------------------------#

    #--- Function: Train a new or existing model ---#
    def train_model(self, model):
        # Build and compile the LSTM, if needed
        if (model == None):
            model = Sequential()
            model.add(LSTM(200, return_sequences=True, input_shape=(self.time_step, 1)))
            model.add(LSTM(200))
            model.add(Dense(128))
            model.add(Dense(32))
            model.add(Dense(8))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(self.X, self.y, epochs=self._epochs, batch_size=64)
        self.last_update = self.last_close()
        print("Model is trained!")
        return model
    #-----------------------------------------------#

    #--- Function: Show how model mirrors actual data ---#
    def mirror_data(self):
        mirror = self._model.predict(self.X)
        mirror = self.scaler.inverse_transform(mirror)
        return mirror
    #-----------------------------------------------#

    #--- Function: Predict price over future given days ---#
    def make_prediction(self, days=_prediction_len):
        # Make sure we have data
        if self._scaled_data is None:
            self.preprocess()
        
        # Prepare enough data for one prediction
        multi_day_data = self._scaled_data[-self.time_step:]
        X_predict = multi_day_data[:self.time_step].reshape(1, self.time_step, 1)

        # Create an array with the first index at last known close price
        prediction = []
        prediction.append(self.orig_data[len(self.orig_data) - 1][0])

        # Predict prices iteratively
        for i in range(days):
            # Predict the next day's price
            scaled_price = self._model.predict(X_predict)
            actual_price = self.scaler.inverse_transform(scaled_price.reshape(-1, 1))
            prediction.append(actual_price[0, 0])

            # Update the input sequence for the next prediction
            multi_day_data = np.append(multi_day_data, scaled_price)
            X_predict = multi_day_data[-self.time_step:].reshape(1, self.time_step, 1)
        
        self.last_pred = prediction
        return prediction
    #------------------------------------------------------#

    #--- Function: Get the last market close date ---#
    def last_close(self):
        # Get the current time & day
        now = datetime.now(pytz.timezone('US/Eastern'))
        today = now.date()

        # If the market is closed, return today
        market_close_time = now.replace(hour=16, minute=0, second=0)
        market_closed = market_close_time <= now
        if market_closed:
            return today

        # Get data for last several closes and capture date
        two_weeks = now - timedelta(days=10)
        minidata = yf.download(self.ticker, start=two_weeks, end=today)
        minidata.reset_index(inplace=True)
        minidata = minidata['Date']

        # Get 'yesterday': the last market close before today
        yesterday = minidata[len(minidata) - 1] # Last market close
        yesterday = yesterday.to_pydatetime().date()
        return yesterday
    #---------------------------------------------------#

    #--- Function: Check if the model needs to be updated ---#
    def needs_update(self):
        # If the model is None, it needs to be updated
        if self._model is None or self.last_update.date() < self.last_close():
            return True

        # Otherwise, no update needed
        return False
#------------------------------------------------------------#

    #--- Function: Train the model on the latest closing price ---#
    def train(self, epochs, mse_threshold=0):
        global model
        # model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
        self.preprocess()

        # Loop until threshold either threshold is met or epochs exausted
        counter = 0
        mse_value = 999999   # Stupidly high error value guarantees one loop
        while (mse_value > mse_threshold) and (counter < epochs):
            epochs_ = epochs
            # Only do 5 epochs at a time if we're going for threshold
            if mse_threshold > 0:
                epochs_ = 5
            history = self._model.fit(self.X, self.y, epochs=epochs_, batch_size=64)
            mse_values = history.history['loss']
            mse_value = mse_values[-1:][0]
            counter += epochs_
            if mse_threshold > 0:
                if mse_value > mse_threshold:
                    if (counter < epochs):
                        print('MSE value ' + str(round(mse_value, 5)) + ' is inadequate, looping again...')
                    else:
                        print('MSE value ' + str(round(mse_value, 5)) + ' is inadequate but epochs maxed out.')
                else:
                    print('MSE value ' + str(round(mse_value, 5)) + ' is adequate.')

        self.last_update = self.last_close()
    #-------------------------------------------------------------#

    #--- Function: Determine whether to buy or sell stock ---#
    def buy_or_sell(self, prediction):
        last_price = self.orig_data[len(self.orig_data) - 1][0]
        last_predicted = prediction[len(prediction) - 1]
        percent = (last_predicted * 100 / last_price)

        if last_price <= last_predicted:
            percent = percent - 100
            percent = str(f"{percent:.2f}")
            return "<b>Buy</b><br>AIStockHelper says this stock will go up in value by " + percent + "%."
        else:
            percent = 100 - percent
            percent = str(f"{percent:.2f}")
            return "<b>Sell</b><br>AIStockHelper says this stock will go down in value by " + percent + "%."
    #---------------------------------------------------------#