import numpy as np
import yf_interface as yfi
import logging
import os
logging.getLogger('tensorflow').setLevel(logging.ERROR) # Set tf logs to error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Turn off oneDNN custom operations
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    # LSTM Performance Variables
    ticker = None               # MSFT | DAC | AAPL
    time_step = 50              # 50   | 100 |
    _epochs = 10                # 10   | 50  | 100
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
        # Check for valid model
        if model is not None:
            self.ticker = ticker
            self._model = model
            self.last_update = last_update
            self.status = status
            print("Model loaded.")
        else:
            # Get data & train brand-new model
            self.ticker = ticker
            self.preprocess('2025-01-01')  # Get data from start date to last close
            print("Training new model...")
            self._model = self._create_model(model)
    #------------------------------#

    # TODO remove after testing
    #--- Function: Create the dataset for LSTM ---#
    # def create_dataset(self, data):
    #     X, y = [], []
    #     for i in range(len(data) - self.time_step - 1):
    #         X.append(data[i:(i + self.time_step), 0])
    #         y.append(data[i + self.time_step, 0])
    #     return np.array(X), np.array(y)
    #---------------------------------------------#

    #--- Function: Preprocess the latest data ---#
    def preprocess(self, end_date=None):
        # Get the latest close prices
        orig_data = None
        # If no end date is provided, use the last close date
        if end_date is None:
            orig_data = yfi.get_close_prices(self.ticker, self._start_date)
        else:
            orig_data = yfi.get_close_prices(self.ticker, self._start_date, end_date)
        self.orig_data = orig_data.reshape(-1, 1)    # Reshape into a 2d array: [[1], [2], [3]]
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self._scaled_data = self.scaler.fit_transform(self.orig_data)

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

        # TODO remove after testing
        # Train model
        # model.fit(self.X, self.y, epochs=self._epochs, batch_size=64)
        # self.last_update = yfi.last_close()
        print("Model has been created, but not trained yet.")
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
        if self._model is None or self.last_update.date() < yfi.last_close():
            return True

        # Otherwise, no update needed
        return False
    #------------------------------------------------------------#

    #--- Function: Train the model on the latest closing price ---#
    def train(self, epochs, end_date=None, mse_threshold=0):
        global model
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
            return (True, "<b>Buy</b><br>AIStockHelper says this stock will go up in value by " + percent + "%.")
        else:
            percent = 100 - percent
            percent = str(f"{percent:.2f}")
            return (False, "<b>Sell</b><br>AIStockHelper says this stock will go down in value by " + percent + "%.")
    #---------------------------------------------------------#