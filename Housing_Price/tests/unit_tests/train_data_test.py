from housing_scripts import train
import unittest
import os

r_path = os.path.dirname(os.path.dirname(os.getcwd()))


class Test(unittest.TestCase):
    def test_train_data(self):
        path = os.path.join(r_path, r"data\processed")
        output_path = os.path.join(r_path, r"artifacts")
        self.assertTrue(
            train.train_data(path, output_path),
            "Test Failed! some issue in function run ",
        )


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
