import numpy as np
import pandas as pd
import tensorflow as tf
import tomli
import matplotlib.pyplot as plt

from common.datamodel1 import make_labelled_dataset as make_model1_dataset
from common.datamodel2 import convert_text_file_to_list
from common.datamodel2 import make_labelled_dataset as make_model2_dataset


def main() -> None:
    with open("model1.toml", "rb") as f:
        conf1 = tomli.load(f)

    with open("model2.toml", "rb") as f:
        conf2 = tomli.load(f)

    # get deep learning model predictions on the test set
    test_set_raw_signal = make_model1_dataset(**conf1["data"]["test"])
    model1 = tf.keras.models.load_model(conf1["artefacts"]["model"])
    y_hat = np.argmax(model1.predict(test_set_raw_signal), axis=-1)

    # create mapping to translate label to activity
    labels = convert_text_file_to_list(conf2["data"]["test"]["labels_desc_file"])
    labels_mapping = {i: label[2:] for i, label in enumerate(labels)}

    # build a dataset made of the base features that contains only the most common
    # error (SITTING predicted as STANDING) as well as the corresponding correct cases
    cases_of_interest = [
        "SITTING predicted as SITTING",
        "SITTING predicted as STANDING",
    ]
    df_to_plot = (
        make_model2_dataset(**conf2["data"]["test"])[conf2["features"]["base"] + ["y"]]
        .assign(
            y_hat=lambda x: y_hat,
            true_label=lambda x: list(map(labels_mapping.get, x.y)),
            pred_label=lambda x: list(map(labels_mapping.get, x.y_hat)),
            case=lambda x: x.true_label + " predicted as " + x.pred_label,
        )
        .query("case in @cases_of_interest")
        .drop(["y", "y_hat", "true_label", "pred_label"], axis=1)
    )

    # project each record in the coordinates space in order to
    # spot consistent differences in trajectories between the two cases
    plt.figure(figsize=(24, 6))
    plt.xticks(rotation=90, fontsize=7)
    plot = pd.plotting.parallel_coordinates(
        df_to_plot, "case", color=("#4ECDC4", "#C7F464")
    )
    plot.get_legend().remove()
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig("report/parallel_coordinates.png")


if __name__ == "__main__":
    main()
