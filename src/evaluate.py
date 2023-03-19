import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tomli
from sklearn.metrics import accuracy_score, confusion_matrix

from common.datamodel1 import get_labels
from common.datamodel1 import make_labelled_dataset as make_model1_dataset
from common.datamodel2 import convert_text_file_to_list
from common.datamodel2 import make_labelled_dataset as make_model2_dataset


def _pretty_confusion_matrix(
    y: list[int], y_hat: list[int], labels_desc_file: str
) -> str:
    """Turn a confusion matrix into a printable string"""
    labels = convert_text_file_to_list(labels_desc_file)
    labels_mapping = {i: label for i, label in enumerate(labels)}
    true_labels = [labels_mapping[ground_truth] for ground_truth in y]
    pred_labels = [labels_mapping[pred] for pred in y_hat]
    df_conf_matrix = pd.DataFrame(
        confusion_matrix(true_labels, pred_labels), index=labels, columns=labels
    )
    return df_conf_matrix.to_string()


if __name__ == "__main__":
    match sys.argv[1]:
        case "deep-learning":
            with open("model1.toml", "rb") as f:
                conf = tomli.load(f)

            model = tf.keras.models.load_model(conf["artefacts"]["model"])
            test_set = make_model1_dataset(**conf["data"]["test"])
            y = get_labels(test_set)
            y_hat = np.argmax(model.predict(test_set), axis=-1)

        case "non-deep-learning":
            with open("model2.toml", "rb") as f:
                conf = tomli.load(f)

            model = joblib.load(conf["artefacts"]["model"])
            test_set = make_model2_dataset(**conf["data"]["test"])
            y = test_set.y
            y_hat = model.predict(test_set)

        case _:
            raise ValueError(
                'The first parameter for this script should be "deep-learning" or "non-deep-learning".'
            )

    accuracy = accuracy_score(y, y_hat)
    conf_matrix_str = _pretty_confusion_matrix(
        y, y_hat, conf["data"]["test"]["labels_desc_file"]
    )

    print()
    print(f"Model accuracy {accuracy:.2%}")
    print()
    print(conf_matrix_str)
    print()
