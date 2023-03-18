from functools import partial
from pathlib import Path
from typing import Callable

import tensorflow as tf


def _parse_raw_ts_file(file: str) -> tf.data.Dataset:
    return (
        tf.data.TextLineDataset(file)
        .map(partial(tf.strings.regex_replace, pattern="\\s\\s+", rewrite=" "))
        .map(tf.strings.split)
        .map(tf.strings.to_number)
    )


def _parse_folder(parser: Callable, folder: str) -> tuple[tf.data.Dataset]:
    return tuple(parser(file) for file in sorted(Path(folder).glob("*")))


def _keep_subjects(subjects: list[int]) -> Callable:
    def _inner_func(ts, label, subject):
        return subject in subjects

    def _tf_py_func(ts, label, subject):
        return tf.py_function(_inner_func, (ts, label, subject), tf.bool)

    return _tf_py_func


# TODO:
# - add test for format
# - add test for split
def make_labelled_dataset(
    feat_folder, labels_file, subjects_file, batch_size, buffer_size, subjects_to_keep
):
    univariate_ts = _parse_folder(_parse_raw_ts_file, feat_folder)
    multivariate_ts = tf.data.Dataset.zip(univariate_ts).map(
        lambda *t: tf.stack(t, axis=-1)
    )
    labels = tf.data.experimental.CsvDataset(
        labels_file, record_defaults=[tf.int32]
    ).map(lambda x: x - 1)
    subjects = tf.data.experimental.CsvDataset(
        subjects_file, record_defaults=[tf.int32]
    )
    return (
        tf.data.Dataset.zip((multivariate_ts, labels, subjects))
        .filter(_keep_subjects(subjects_to_keep))
        .map(lambda ts, label, _: (ts, label))
        .shuffle(buffer_size)
        .batch(batch_size)
    )
