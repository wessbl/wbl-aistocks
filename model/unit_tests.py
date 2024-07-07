import unittest
from lstm_model import LSTMModel

class TestModels(unittest.TestCase):

    def lstm_ctor(self):
        model = LSTMModel('AAPL')
    
    def lstm_last_close(self):
        model = LSTMModel('AAPL')
