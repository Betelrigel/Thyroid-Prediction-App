import unittest
import pandas as pd
from src.feature_engineering import engineer_features

class TestFeatureEngineering(unittest.TestCase):
    def test_engineer_features(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        transformed_df = engineer_features(df)
        self.assertEqual(transformed_df.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
