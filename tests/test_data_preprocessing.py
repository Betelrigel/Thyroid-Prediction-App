import unittest
import pandas as pd
from src.data_preprocessing import clean_data

class TestDataPreprocessing(unittest.TestCase):
    def test_clean_data(self):
        df = pd.DataFrame({'A': [1, None, 3], 'B': [None, 2, 3]})
        clean_df = clean_data(df)
        self.assertFalse(clean_df.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
