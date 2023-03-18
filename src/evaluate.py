from pathlib import Path

import tensorflow as tf
import tomli

from common.data import make_labelled_dataset


def main():
    model = tf.keras.models.load_model("model")

    with open("config.toml", "rb") as f:
        conf = tomli.load(f)

    subjects = set(
        int(s) for s in Path(conf["data"]["test_subjects_file"]).read_text().split()
    )

    test_set = make_labelled_dataset(
        conf["data"]["test_feat_folder"],
        conf["data"]["test_labels_file"],
        conf["data"]["test_subjects_file"],
        subjects_to_keep=list(subjects),
        **conf["make_labelled_dataset"]
    )

    model.evaluate(test_set)


if __name__ == "__main__":
    main()
