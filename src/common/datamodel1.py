from functools import partial
from pathlib import Path
from typing import Callable, Optional, Any
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
    feat_folder: str,
    labels_file: str,
    subjects_file: str,
    batch_size: int,
    seed: Optional[int]=None,
    buffer_size: Optional[int]=None,
    subjects_to_keep: Optional[list[int]]=None,
    **kwargs: Any
):
    if seed and not buffer_size:
        raise ValueError('The parameter buffer_size is required for shuffling the dataset.')

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

    dataset = tf.data.Dataset.zip((multivariate_ts, labels, subjects))

    if subjects_to_keep:
        dataset = dataset.filter(_keep_subjects(subjects_to_keep))

    dataset = dataset.map(lambda ts, label, _: (ts, label))

    if seed:
        dataset = dataset.shuffle(buffer_size, seed=seed)

    return dataset.batch(batch_size)


def get_labels(labelled_dataset: tf.data.Dataset) -> list[int]:
    labels = labelled_dataset.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x[-1])).as_numpy_iterator()
    return list(labels)

def get_subjects(file: str) -> set[int]:
    return set(int(s) for s in Path(file).read_text().split())
