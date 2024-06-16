import unittest
import requests

# before running tests make sure that microservice is running

class TestMicroservice(unittest.TestCase):
    def test_gb_correct_length(self):
        res = requests.get("http://127.0.0.1:8000/generate?genre=pop&num_tracks=25&model=GB")
        text = res.text[1:-1] # remove brackets from encapsulating list
        self.assertEqual(text.count('['), 25)

    def test_nn_correct_length(self):
        res = requests.get("http://127.0.0.1:8000/generate?genre=metal&num_tracks=30&model=NN")
        text = res.text[1:-1] # remove brackets from encapsulating list
        self.assertEqual(text.count('['), 30)


if __name__ == '__main__':
    unittest.main()