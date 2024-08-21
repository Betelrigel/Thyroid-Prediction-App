import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_model

class TestEvaluate(unittest.TestCase):
    def test_evaluate_model(self):
        model = RandomForestClassifier()
        X_test = pd.DataFrame({'A': [1, 2], 'B': [4, 5]})
        y_test = [0, 1]
        model.fit(X_test, y_test)
        
        try:
            report = evaluate_model(model, X_test, y_test)
            self.assertIsInstance(report, str)
        except Exception as e:
            self.fail(f"Evaluation failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
