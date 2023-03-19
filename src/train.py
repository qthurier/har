import random

import tensorflow as tf
import tomli

from common.data import get_subjects, make_labelled_dataset
from model.learner import Classifier


def main():
    # TODO: check balance
    with open("config.toml", "rb") as f:
        conf = tomli.load(f)

    random.seed(conf["data"]["seed"])
    subjects = get_subjects(conf["data"]["train"]["subjects_file"])
    validation_subjects = random.sample(subjects, conf["data"]["n_validation_subjects"])
    training_subjects = [s for s in subjects if s not in validation_subjects]

    validation_set = make_labelled_dataset(
        subjects_to_keep=validation_subjects,
        **conf["data"]["train"],
        **conf["make_labelled_dataset"]
    )

    training_set = make_labelled_dataset(
        subjects_to_keep=training_subjects,
        **conf["data"]["train"],
        **conf["make_labelled_dataset"]
    )

    classifier = Classifier(**conf["Classifier"])
    classifier.compile(
        optimizer="sgd",
        loss="sparse_categorical_crossentropy",
        metrics="sparse_categorical_accuracy",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", restore_best_weights=True, patience=conf["training"]["patience"]
    )
    classifier.build((None, 128, 9))
    classifier.fit(
        training_set,
        validation_data=validation_set,
        callbacks=[early_stopping],
        epochs=conf["training"]["max_epochs"],
    )

    classifier.save(conf["artefacts"]["deep_learning_model"])


if __name__ == "__main__":
    main()
