# FutureStock AI

A Flask-based LSTM application that predicts future stock prices for selected tickers.

<img src="screenshots/futurestock-ss.png" alt="FutureStock AI Screenshot" height="500">

## Description

This application trains LSTM models on selected stock tickers and continuously updates them. With every new closing price, the app loads a pre-trained model from an SQLite database. The model is trained on the fresh data, a new five-day forecast is generated, and both are saved. Results are given as percentages and graphs.

### Results

One Buy/Sell recommendation is provided every day based on the predicted change.

The prediction graph shows historical prices in blue and the most recent forecast in orange, with a dotted line marking the boundary between actual market prices and predicted prices.

Buy/sell recommendation accuracy notes how accurately each prediction matches the next day's actual price movement.

Simulated return shows the percentage loss/gain of following each Buy/Sell recommendation.

Mean Absolute Percentage Error (MAPE) is a statistical metric that measures the average size of prediction errors as a percentage - lower is better.

The Model Against Actual Price graph shows the next-day predictions overlaid on actual prices across the entire training set used by the model, demonstrating how closely the model's predictions align with the stock price.

> **Note:** This is a portfolio project demonstrating AI integration and should not be interpreted as financial advice.

## Getting Started

### Dependencies

* Python 3
* `requirements.txt` includes all necessary packages such as Flask, TensorFlow, Pandas, etc.

### Installing

* Install git, python3, python3-pip, pip3 as needed
* Clone the project to your desired directory
* Create the virtual environment. Go into the project folder and run:
``` 
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Executing program

* Open a terminal to the root directory
* To run the app in development mode:
``` 
python app.py
```
This will launch the app locally at `http://127.0.0.1:5000/`.


## Help

* For a basic walkthrough, see my [Jupyter Notebook Version](https://colab.research.google.com/drive/1z96VjkJXcIOQ6KdNjEPjhmxKKfLd7FLH).
* The app starts fresh when `model.db` is deleted: `static/models/model.db`.
* To tweak performance or accuracy, adjust the global variables in `model/lstm_model.py`. For example, lowering `epochs` speeds up training but reduces accuracy.

## Author

**Wess Lancaster**  
[LinkedIn ↗](https://linkedin.com/in/wessbl)  
wess.lancaster@gmail.com

## Version History
* 0.7 DB Overhaul
    * Massive back-end improvements not yet available in the UI: database overhauled, predictions tracked, and the updater now calculates daily APE/MAPE and profit simulation values
    * Front end now relies entirely on database data; no model creation occurs in the UI
    * DBI and YFI are instantiated at startup and their objects are passed to other classes as needed
    * YFI caches all stock prices on load and serves slices of it on demand
    * Fixed caching issues that previously showed incorrect predictions on past days
    * Models are initially trained on every day since their last update (or from a fallback start date for new models)
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/6e0d5d232b39b50cb865c77d185daf6909aeb054)
* 0.6 Updater
    * Program has an updater file that can be run by a service automatically
    * Users no longer need to wait for a model to train
    * Models are now stored with a status for update handling
    * If user accesses a ticker during an update a message is displayed above the older results
    * Removed browser caching to ensure that the latest prediction is always shown
    * Added basic formatting for prediction message (bold and new line)
    * Removed "Not loading?" Message
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/2406748f5aac82f328ce579fc554bb37e5ea3610)
* 0.5 Front-End Rework
    * Images now resize with smaller screens/windows
    * Added white paper and repo links
    * Added hideable "Not loading?" message
    * Decreased prediction length due to slow server response
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/5740fac657a7a16181d3a19ea7f43b089d096ad2)
* 0.4 General Updates
    * Squashed bug that prevented model updates
    * Added db_interface.py - currently only for admin operations
    * Directory cleanup
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/22f1e557d6ba796af350a90c0b23e42befec3ae0)
* 0.3 General Updates
    * General directory cleanup, removal of debug prints
    * Lowered epochs due to slow server response
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/678685f3d3f6fe2298b1375f311f48b0a9492b44)
* 0.2 General Updates
    * All models held in a dictionary for immediate response time
    * Constant prediction & image generation prevented
    * Wrapper class added around lstm_model.py
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/2fb715f51e4b70cdd910bbfd11f17d2433b050c5)
* 0.1
    * Initial Release