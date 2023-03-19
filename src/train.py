import random
import sys

import tomli
import joblib
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from common.datamodel1 import make_labelled_dataset as make_model1_dataset
from common.datamodel1 import get_subjects
from common.datamodel2 import make_labelled_dataset as make_model2_dataset
from model.model1 import Classifier as DeepLearningClassifier
from model.model2 import Classifier as RandomForestClassifier


def train_model1(conf: dict) -> None:
    random.seed(conf["data"]["train"]["seed"])
    subjects = get_subjects(conf["data"]["train"]["subjects_file"])
    validation_subjects = random.sample(
        subjects, conf["data"]["train"]["n_validation_subjects"]
    )
    training_subjects = [s for s in subjects if s not in validation_subjects]

    validation_set = make_model1_dataset(
        subjects_to_keep=validation_subjects,
        **conf["data"]["train"],
    )

    training_set = make_model1_dataset(
        subjects_to_keep=training_subjects,
        **conf["data"]["train"],
    )

    classifier = DeepLearningClassifier(**conf["Classifier"])
    classifier.compile(
        loss=conf["training"]["loss"],
        metrics=conf["training"]["metric"],
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


def train_model2(conf: dict) -> None:
    training_set = make_model2_dataset(**conf["data"]["train"])
    classifier = RandomForestClassifier(
        base_features=conf["features"]["base"],
        time_domain_groups=conf["features"]["time_domain_groups"],
        freq_domain_groups=conf["features"]["freq_domain_groups"],
        max_extra_feat=conf["Classifier"]["max_extra_feat"],
        max_iter=conf["training"]["max_iter"],
        seed=conf["training"]["seed"],
    )
    grid = conf["training"]["hyperparameter_grid"]
    best_classifier = GridSearchCV(
        classifier,
        grid,
        scoring=conf["training"]["metric"],
        cv=conf["training"]["n_folds"],
        n_jobs=-1,
    )
    best_classifier.fit(training_set, training_set.y)
    Path(conf["artefacts"]["model"]).parent.mkdir(parents=True)
    joblib.dump(best_classifier, conf["artefacts"]["model"])


if __name__ == "__main__":
    match sys.argv[1]:
        case "deep-learning":
            with open("model1.toml", "rb") as f:
                conf = tomli.load(f)

            train_model1(conf)

        case "non-deep-learning":
            with open("model2.toml", "rb") as f:
                conf = tomli.load(f)

            train_model2(conf)

        case _:
            raise ValueError(
                'The first parameter for this script should be "deep-learning" or "non-deep-learning".'
            )
