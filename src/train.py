import random
from pathlib import Path

import tensorflow as tf
import tomli

from common.data import make_labelled_dataset
from model.learner import Classifier


def main():
    # TODO: check balance
    with open("config.toml", "rb") as f:
        conf = tomli.load(f)

    subjects = set(
        int(s) for s in Path(conf["data"]["train_subjects_file"]).read_text().split()
    )
    validation_subjects = random.sample(subjects, conf["data"]["n_validation_subjects"])
    training_subjects = [s for s in subjects if s not in validation_subjects]

    validation_set = make_labelled_dataset(
        conf["data"]["train_feat_folder"],
        conf["data"]["train_labels_file"],
        conf["data"]["train_subjects_file"],
        subjects_to_keep=validation_subjects,
        **conf["make_labelled_dataset"]
    )

    training_set = make_labelled_dataset(
        conf["data"]["train_feat_folder"],
        conf["data"]["train_labels_file"],
        conf["data"]["train_subjects_file"],
        subjects_to_keep=training_subjects,
        **conf["make_labelled_dataset"]
    )

    classifier = Classifier(**conf["Classifier"])
    classifier.compile(
        optimizer="sgd",
        loss="sparse_categorical_crossentropy",
        metrics="sparse_categorical_accuracy",
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3, restore_best_weights=True
    )
    classifier.fit(
        training_set, validation_data=validation_set, epochs=10, callbacks=[callback]
    )

    classifier.save("keras_artefacts")


if __name__ == "__main__":
    main()
