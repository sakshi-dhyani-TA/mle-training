import unittest
import os

r_path = os.path.dirname(os.path.dirname(os.getcwd()))


class Test(unittest.TestCase):
    def test_model_file(self):
        path = os.path.join(r_path, r"artifacts\housing_model.pkl")
        self.assertTrue(os.path.exists(path), "Model not exists! Test Failed ")


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
