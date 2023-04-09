import os
import unittest

r_path = os.path.dirname(os.path.dirname(os.getcwd()))


class Test(unittest.TestCase):
    def test_score_file(self):
        path = os.path.join(r_path, r"Score\Evaluation_Metrics.txt")
        self.assertTrue(
            os.path.exists(path), "Evaluation Metrics File not exists! Test Failed "
        )


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
