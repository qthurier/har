import random
import sys

import tomli
from joblib import dump
from keras.callbacks import EarlyStopping

from common.data import get_subjects, make_labelled_dataset
from common.datamodel2 import make_labelled_dataset as make_model2_dataset
from model.learner import Classifier
from model.model2 import Classifier as TreeBasedClassifier


def train_model1(conf: dict):

    random.seed(conf["data"]["train"]["seed"])
    subjects = get_subjects(conf["data"]["train"]["subjects_file"])
    validation_subjects = random.sample(
        subjects, conf["data"]["train"]["n_validation_subjects"]
    )
    training_subjects = [s for s in subjects if s not in validation_subjects]

    validation_set = make_labelled_dataset(
        subjects_to_keep=validation_subjects,
        **conf["data"]["train"],
    )

    training_set = make_labelled_dataset(
        subjects_to_keep=training_subjects,
        **conf["data"]["train"],
    )

    classifier = Classifier(**conf["Classifier"])
    classifier.compile(
        optimizer="sgd",
        loss="sparse_categorical_crossentropy",
        metrics="sparse_categorical_accuracy",
    )
    early_stopping = EarlyStopping(
        monitor="loss", restore_best_weights=True, patience=conf["training"]["patience"]
    )
    classifier.fit(
        training_set,
        validation_data=validation_set,
        callbacks=[early_stopping],
        epochs=conf["training"]["max_epochs"],
    )

    classifier.save(conf["artefacts"]["model"])


def train_model2(conf: dict):
    training_set = make_model2_dataset(**conf["data"]["train"])
    classifier = TreeBasedClassifier(conf["features"]["base"])
    classifier.fit(training_set, training_set.y)
    dump(classifier, conf["artefacts"]["model"])


if __name__ == "__main__":
    # TODO: check balance
    if sys.argv[1] == "deep-learning":

        with open("model1.toml", "rb") as f:
            conf = tomli.load(f)

        train_model1(conf)

    elif sys.argv[1] == "non-deep-learning":

        with open("model2.toml", "rb") as f:
            conf = tomli.load(f)

        train_model2(conf)

    else:
        raise ValueError(
            'The first parameter for this script should be "deep-learning" or "non-deep-learning".'
        )
