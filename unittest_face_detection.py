import unittest
import tempfile
import os
from script import read_data


class TestFaceDetection(unittest.TestCase):
    def test_read_data(self):
        ground_truth = {
            'image1.jpg': 3,
            'image2.jpg': 1,
            'image3.jpg': 0,
        }

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            for image, count in ground_truth.items():
                temp.write(f'{image} {count}\n'.encode())

        observed_output = read_data(temp.name)
        os.unlink(temp.name)

        self.assertDictEqual(ground_truth, observed_output)


if __name__ == '__main__':
    unittest.main()