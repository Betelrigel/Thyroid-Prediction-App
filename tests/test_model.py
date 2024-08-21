import unittest
import pandas as pd
from src.model import train_model

class TestModel(unittest.TestCase):
    def test_train_model(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [4, 5, 6, 7],
            'target': [0, 1, 0, 1]
        })
        model, X_test, y_test = train_model(df)
        self.assertEqual(len(X_test), 1)

if __name__ == '__main__':
    unittest.main()
