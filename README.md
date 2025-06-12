# FutureStock AI

A Flask web app that predicts future stock prices.

## Description

This application trains LSTM models on a set of stocks, keeping them updated with every new closing price. **This project is merely a demonstration of AI and should not be taken as financial advice.**

When a user selects a stock ticker, the application loads the model trained on that ticker from an SQLite database, along with its most recent predictions. The model is updated and makes a prediction if the market has closed since its last prediction, images are generated, and everything is saved.

## Getting Started

### Dependencies

* python3 to create virtual environments and run the program
* Flask, yfinance, sqlite3, matplotlib, numpy, pandas, tensorflow, keras, sklearn
* The full list of requirements is in the requirements.txt file, along with the versions for each package. This makes installation much easier from a terminal.

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
* To run a local development server, use:
``` 
python app.py
```

## Help

* For a basic demonstration of how the code works, check out my [Jupyter Notebook Version](https://colab.research.google.com/drive/1z96VjkJXcIOQ6KdNjEPjhmxKKfLd7FLH)
* The program starts from scratch when there is no database file. You can delete it at anytime: root/static/models/model.db
* To tweak the models more accurate or lower the loading times, make changes to the global variables in root/model/lstm_model.py. For example, lowering the epochs will decrease both loading times and accuracy in general.
* Feel free to email me at wess.lancaster@gmail.com

## Author

Wess Lancaster  
[LinkedIn](https://linkedin.com/in/wessbl)

## Version History

* 0.5
    * Images now resize with smaller screens/windows
    * Added white paper and repo links
    * Added hideable "Not loading?" message
    * Decreased prediction length due to slow server response
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/5740fac657a7a16181d3a19ea7f43b089d096ad2)
* 0.4
    * Squashed bug that prevented model updates
    * Added db_interface.py - currently only for admin operations
    * Directory cleanup
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/22f1e557d6ba796af350a90c0b23e42befec3ae0)
* 0.3
    * General directory cleanup, removal of debug prints
    * Lowered epochs due to slow server response
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/678685f3d3f6fe2298b1375f311f48b0a9492b44)
* 0.2
    * All models held in a dictionary for immediate response time
    * Constant prediction & image generation prevented
    * Wrapper class added around lstm_model.py
    * See [commit change](https://github.com/wessbl/wbl-aistocks/commit/2fb715f51e4b70cdd910bbfd11f17d2433b050c5)
* 0.1
    * Initial Release