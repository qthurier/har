import tensorflow as tf
import tomli

from common.data import get_subjects, make_labelled_dataset


def main():
    with open("config.toml", "rb") as f:
        conf = tomli.load(f)

    model = tf.keras.models.load_model(conf["artefacts"]["deep_learning_model"])
    subjects = get_subjects(conf["data"]["test"]["subjects_file"])
    test_set = make_labelled_dataset(
        subjects_to_keep=list(subjects),
        **conf["data"]["test"],
        **conf["make_labelled_dataset"]
    )

    model.evaluate(test_set)


if __name__ == "__main__":
    main()
