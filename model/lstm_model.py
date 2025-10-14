import numpy as np
from model.yf_interface import YFInterface
import logging
import os
logging.getLogger('tensorflow').setLevel(logging.ERROR) # Set tf logs to error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Turn off oneDNN custom operations
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    # LSTM Performance Variables
    ticker = None               # MSFT | DAC | AAPL
    time_step = 50              # 50   | 100 |
    _epochs = 50                # 10   | 50  | 100
    _update_epoch = 1           # how many epochs for an update
    _prediction_len = 5         # how many days to predict
    _start_date = '2017-01-01'  # Initial training start date
    last_update = None      # Last update as a date 'YYYY-MM-DD'
    _model = None
    orig_data = None
    _scaled_data = None
    prediction = None
    recommendation = None
    _yf = None

    X = np.array([])
    scaler = MinMaxScaler(feature_range=(0,1))

    #--- Constructor ---#
    def __init__(self, ticker, model=None, last_update=None, status=None, yf=None):
        # Check for valid model
        if model is not None:
            self.ticker = ticker
            self._model = model
            self.last_update = last_update
            self.status = status
        else:
            # Get data & train brand-new model
            self.ticker = ticker
            self.status = 'new'
            self.last_update = '2025-09-01'
            self._model = self._create_model(model)
        self._yf = yf
    #------------------------------#

    #--- Function: Preprocess the latest data ---#
    def preprocess(self, end_date=None):
        # Get the latest close prices
        orig_data = None
        # If no end date is provided, use the last close date
        if end_date is None:
            orig_data = self._yf.get_close_prices(self.ticker, self._start_date)
        else:
            orig_data = self._yf.get_close_prices(self.ticker, self._start_date, end_date)
        self.orig_data = orig_data.reshape(-1, 1)    # Reshape into a 2d array: [[1], [2], [3]]
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self._scaled_data = self.scaler.fit_transform(self.orig_data)
        if np.isnan(self._scaled_data).any():
            raise ValueError(f"Scaled data for {self.ticker} contains NaNs on {end_date}")

        # Create Datasets to feed LSTM
        X, y = [], []
        for i in range(len(self._scaled_data) - self.time_step - 1):
            X.append(self._scaled_data[i:(i + self.time_step), 0])
            y.append(self._scaled_data[i + self.time_step, 0])
        self.X, self.y = np.array(X), np.array(y)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
    #---------------------------------------------#

    #--- Function: Set model properties and compile ---#
    def _create_model(self, model):
        # TODO do some quick testing to find which values work best for each ticker
        # Build and compile the LSTM, if needed
        if (model == None):
            model = Sequential()
            model.add(Input(shape=(self.time_step, 1)))
            model.add(LSTM(200))
            model.add(Dense(128))
            model.add(Dense(32))
            model.add(Dense(8))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

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

    #--- Function: Check if the model needs to be updated ---#
    def needs_update(self):
        # If the model is None, it needs to be updated
        if self._model is None or self.last_update is None or self.last_update < self._yf.last_close():
            # TODO probably not needed
            # print(f"Needs Update: ", end='')
            # if self._model is None:
            #     print("Model is None.")
            # elif self.last_update is None:
            #     print("Last update is None.")   #TODO make sure last_update is being set!
            # else:
            #     print(f"Last update {self.last_update} is before {yfi.last_close()}.")
            return True

        # Otherwise, no update needed
        return False
    #------------------------------------------------------------#

    #--- Function: Train the model up to given date ---#
    def train(self, epochs, end_date=None, mse_threshold=0):
        if end_date is None:
            self.preprocess()
        else:
            self.preprocess(end_date)

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
        self.last_update = self._yf.last_close()
    #-------------------------------------------------------------#

    #--- Function: Determine whether to buy or sell stock ---#
    def percentage_change(self, prediction):
        last_price = self.orig_data[len(self.orig_data) - 1][0]
        last_predicted = prediction[len(prediction) - 1]
        ratio = (last_predicted / last_price)
        percent = (ratio - 1) * 100
        return percent
    #---------------------------------------------------------#