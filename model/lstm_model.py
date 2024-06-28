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
    ticker = 'AAPL'         # MSFT | DAC | AAPL
    years = 10              #  5  | 10  |
    time_step = 75          # 50  | 100 |
    lstm_unit = 75          # 50  | 64  | 100-200, choose one layer only
    dnse_unit = 32          #  1  | 64  | choose < 5 layers no more than 128
    batch = 64              # 32  | 64  |
    epochs = 20             # 10  | 50  | 100
    update_epoch = 3        # how many epochs for an update

    # Other Variables
    save_model = True       # True False - just helps prevent time while editing
    verbose = True          # Display outputs
    offset = 0              # Slide predictions to match real values
    prediction_len = 5      # how many days to predict
    last_pred = None        # Model's last prediction
    last_mirror = None
    result = ''
    model_filename = None
    db_path = None
    arch_filename = 'model_architecture.png'
    start_date = '2014-01-01'
    last_update = pd.Timestamp('2023-12-31')
    model = None
    orig_data = None
    scaled_data = None

    X = np.array([])
    scaler = MinMaxScaler(feature_range=(0,1))

    #--- Constructor ---#
    def __init__(self, ticker, mdl_dir):
        if self.verbose:
            print('\n\n\nTicker:\t', ticker)
            print('-------------')
        
        # Set global vars
        self.ticker = ticker
        self.model_filename = mdl_dir + 'lstm-' + ticker + '.keras'
        self.db_path = mdl_dir + 'models.db'

        # First, try to load the model
        try:
            self.model, self.last_update, result = self.load(self.ticker)
            
            # Check for updates
            updated, expl = self.update_model()
            if self.verbose: print("\nModel Updated:\t", updated, "\nExplanation:\t", expl)

            # Process data if needed
            if (self.orig_data is None) or (len(self.orig_data) == 0):
                self.preprocess(yf.download(self.ticker, start=self.start_date, end=self.last_update))
                if self.verbose: print('Data loaded & processed!')

        except:
            # Train a new model on 10+ years of data
            if self.verbose: print("\nFetching data...")
            last_day = pd.Timestamp.now().date()
            df = yf.download(self.ticker, start=self.start_date, end=last_day)
            last_day = last_day - timedelta(days=1) # yf excludes end date
            self.last_update = last_day
            if (len(df) == 0): raise ValueError("Cannot download from yfinance")
            self.preprocess(df)

            # Train a new model
            if self.verbose: print("Training model...")
            self.model = self.train_model(self.model)

        prediction = self.make_prediction(self.prediction_len)
        self.result = self.buy_or_sell(prediction)
        self.last_mirror = self.mirror_data(self.model)
        self.save(self.ticker, self.model, self.last_update, self.result)
        return
    #------------------------------#

    #--- Function: Create the dataset for LSTM ---#
    def create_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.time_step - 1):
            X.append(data[i:(i + self.time_step), 0])
            y.append(data[i + self.time_step, 0])
        return np.array(X), np.array(y)
    #---------------------------------------------#


    #--- Function: Preprocess data ---#
    def preprocess(self, df):
        # global orig_data, scaler, scaled_data, X, y
        self.orig_data = df['Close'].values
        self.orig_data = self.orig_data.reshape(-1, 1)    # Reshape into a 2d array: [[1], [2], [3]]
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(self.orig_data)

        # Create Datasets to feed LSTM
        self.X, self.y = self.create_dataset(self.scaled_data)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
        return
    #---------------------------------#


    #--- Function: Train a new or existing model ---#
    def train_model(self, model):
        # Build and compile the LSTM, if needed
        if (model == None):
            model = Sequential()
            model.add(LSTM(self.lstm_unit, return_sequences=True, input_shape=(self.time_step, 1)))
            model.add(LSTM(self.lstm_unit))
            model.add(Dense(5)) # Capture weekly cycles, 5 trading days/week
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(self.X, self.y, epochs=self.epochs, batch_size=self.batch)
        if self.verbose: print("Model is trained!")
        return model
    #-----------------------------------------------#

    #--- Function: Save to DB ---#
    def save(self, ticker, model, last_update, result=''):
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
    def load(self, ticker):
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
        return
    #------------------------------#

    #--- Function: Show how model mirrors actual data ---#
    def mirror_data(self, model):
        mirror = model.predict(self.X)[self.offset:]
        mirror = self.scaler.inverse_transform(mirror)
        return mirror
    #-----------------------------------------------#

    #--- Function: Predict price over future given days ---#
    def make_prediction(self, days):
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
