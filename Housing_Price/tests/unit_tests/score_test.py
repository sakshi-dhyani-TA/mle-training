import os
from src import score
import unittest

r_path = os.path.dirname(os.path.dirname(os.getcwd()))


class Test(unittest.TestCase):
    def test_train_data(self):
        input_path = os.path.join(r_path, r"data\processed")
        output_path = os.path.join(r_path, r"Score")
        model_path = os.path.join(r_path, r"artifacts")
        self.assertTrue(
            score.metrics_evaluation(input_path, output_path, model_path),
            "Test Failed! some issue in function run ",
        )


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
