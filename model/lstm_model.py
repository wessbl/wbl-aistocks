import yfinance as yf
import numpy as np
import pandas as pd
import sqlite3
import pytz
from datetime import datetime, timedelta
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    # LSTM Performance Variables
    ticker = None               # MSFT | DAC | AAPL
    _time_step = 75             # 50  | 100 |
    _epochs = 2                 # 10  | 50  | 100      # TODO set to 100
    _update_epoch = 3           # how many epochs for an update
    _prediction_len = 10        # how many days to predict  #TODO set to 5?
    _start_date = '2015-01-01'
    last_update = pd.Timestamp('2023-12-31')
    _model = None
    orig_data = None
    _scaled_data = None
    mirror = None
    prediction = None
    recommendation = None

    X = np.array([])
    scaler = MinMaxScaler(feature_range=(0,1))

    #--- Constructor ---#
    def __init__(self, ticker, model=None):
        # Check for valid model & attempt update
        if model is not None:
            self.ticker = ticker
            self.model = model
        else:
            # Get data & train brand-new model
            self.preprocess()
            print("Training new model...")
            self.model = self.train_model(model)
        
        # Generate outputs now
        self.mirror = self.mirror_data(self.model)
        self.prediction = self.make_prediction()
        self.recommendation = self.buy_or_sell(self.prediction)
    #------------------------------#

    #--- Function: Create the dataset for LSTM ---#
    def create_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self._time_step - 1):
            X.append(data[i:(i + self._time_step), 0])
            y.append(data[i + self._time_step, 0])
        return np.array(X), np.array(y)
    #---------------------------------------------#

    #--- Function: Preprocess the latest data ---#
    def preprocess(self):
        # TODO we have lots of dates flying around here now
        # Get all data from start through last close (yf excludes end date)
        today = pd.Timestamp.now().date()
        print(today)
        print(self.last_close)
        print(today == self.last_close)
        if today == self.last_close:
            df = yf.download(self.ticker, start=self.start_date)
        else: 
            df = yf.download(self.ticker, start=self.start_date, end=today)
        if (len(df) == 0): raise ValueError("Cannot download from yfinance")

        # Create global orig_data, scaler, scaled_data, X, y
        orig_data = df['Close'].values
        self.orig_data = orig_data.reshape(-1, 1)    # Reshape into a 2d array: [[1], [2], [3]]
        scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = scaler.fit_transform(self.orig_data)

        # Create Datasets to feed LSTM
        self.X, self.y = self.create_dataset(self.scaled_data)
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
        model.fit(self.X, self.y, epochs=self.epochs, batch_size=64)
        self.last_update = pd.Timestamp.now().date()
        if self.verbose: print("Model is trained!")
        return model
    #-----------------------------------------------#

    #--- Function: Show how model mirrors actual data ---#
    def mirror_data(self, model):
        mirror = model.predict(self.X)[self.offset:]
        mirror = self.scaler.inverse_transform(mirror)
        return mirror
    #-----------------------------------------------#

    #--- Function: Predict price over future given days ---#
    def make_prediction(self, days=_prediction_len):
        # Make sure we have data
        if self.scaled_data is None:
            self.preprocess(yf.download(self.ticker, start=self.start_date, end=self.last_update))
        
        # Prepare enough data for one prediction
        multi_day_data = self.scaled_data[-self.time_step:]
        X_predict = multi_day_data[:self.time_step].reshape(1, self.time_step, 1)

        # Create an array with the first index at last known close price
        prediction = []
        prediction.append(self.orig_data[len(self.orig_data) - 1][0])

        # Predict prices iteratively
        for i in range(days):
            # Predict the next day's price
            scaled_price = self.model.predict(X_predict)
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

    #--- Function: Train the model on the latest closing price ---#
    def update_model(self):
        # Get the current time & day
        now = datetime.now(pytz.timezone('US/Eastern'))
        today = now.date()

        # If the market is closed and we have already updated, return
        market_close_time = now.replace(hour=16, minute=0, second=0)
        market_closed = market_close_time <= now
        if market_closed & (self.last_update.date() == today):
            return False, "Model is already updated on today's market close."

        # Get 'yesterday': the last market close before today
        two_weeks = now - timedelta(days=10)    #enough data for several closes
        minidata = yf.download(self.ticker, start=two_weeks, end=today)
        minidata.reset_index(inplace=True)
        minidata = minidata['Date']
        yesterday = minidata[len(minidata) - 1] # Last market close
        yesterday = yesterday.to_pydatetime().date()

        # If the market isn't closed but last update was yesterday, return
        if (not market_closed) & (self.last_update.date() == yesterday):
            return False, "Model is already updated to previous market close."

        # Update model with all close prices from original start date
        if (not market_closed):
            today = yesterday
        #TODO downloading from yf might be irrelevant now
        df = yf.download(self.ticker, start=self.start_date, end=today)
        self.preprocess(df)
        self.model.fit(self.X, self.y, epochs=self.update_epoch, batch_size=self.batch)
        self.last_update = today
        dates = 'Updated model: ' + self.start_date + ' through ' + self.last_update.strftime("%Y-%m-%d") + '.'
        return True, dates
    #-------------------------------------------------------------#

    #--- Function: Determine whether to buy or sell stock ---#
    def buy_or_sell(self, prediction):
        last_price = self.orig_data[len(self.orig_data) - 1][0]
        last_predicted = prediction[len(prediction) - 1]
        percent = (last_predicted * 100 / last_price)

        if last_price <= last_predicted:
            percent = percent - 100
            percent = str(f"{percent:.2f}")
            return "Buy: AIStockHelper says this stock will go up in value by " + percent + "%."
        else:
            percent = 100 - percent
            percent = str(f"{percent:.2f}")
            return "Sell: AIStockHelper says this stock will go down in value by " + percent + "%."
    #---------------------------------------------------------#
