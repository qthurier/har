import tensorflow as tf
from common.datamodel1 import make_labelled_dataset
import tomli


class DataModel1(tf.test.TestCase):
    def test_make_labelled_dataset_outputs_expected_shapes(self):
        """Test the assignment requirement regarding batch shapes (ie must be batch size x sequence length x number of features)"""

        with open("model1.toml", "rb") as f:
            conf = tomli.load(f)

        dataset = make_labelled_dataset(**conf["data"]["train"])

        for eg in dataset.take(1):
            actual_shape = eg[0].get_shape()

        expected_shape = tf.constant([conf["data"]["train"]["batch_size"], 128, 9])

        self.assertAllEqual(actual_shape, expected_shape)
