import os
import unittest

from housing_scripts import ingest_data


class Test(unittest.TestCase):
    def test_ingest_data(self):
        r_path = os.path.dirname(os.path.dirname(os.getcwd()))
        path = os.path.join(r_path, r"data\processed")
        self.assertTrue(
            ingest_data.process_data(file_path=path, inp_path=r_path),
            "Test Failed! some issue in function run ",
        )


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
