import pandas as pd
from pathlib import Path
from typing import Optional, Any


def convert_text_file_to_list(file: str) -> list[str]:
    """Turn a single column text file into a list"""
    return [s for s in Path(file).read_text().split("\n") if s]


def make_labelled_dataset(
    feat_file: str,
    headers_file: str,
    labels_file: str,
    seed: Optional[int] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """Create a pandas dataframe containing features and labels, used equally to build training and test sets"""
    feat_cols = convert_text_file_to_list(headers_file)
    labels = pd.read_csv(labels_file, header=None, names=["y"])
    feats = pd.read_csv(feat_file, header=None, delimiter=r"\s+", names=feat_cols)
    dataset = pd.concat(
        [feats, labels - 1], axis=1
    )  # ensure labels start at 0 for convenience
    if seed:
        dataset = dataset.sample(frac=1, random_state=seed)
    return dataset
