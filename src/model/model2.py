from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Any


class FeatureSelector(Pipeline):
    def __init__(
        self,
        block: str,
        columns: list[str],
        max_iter: int,
        max_features: int,
        **kwargs: Any,
    ) -> None:
        self.block = block
        self.columns = columns
        self.max_features = max_features
        self.max_iter = max_iter
        steps = [
            (
                f"{self.block}_slice",
                ColumnTransformer(
                    [("selector", "passthrough", self.columns)], remainder="drop"
                ),
            ),
            (
                f"{self.block}_select",
                SelectFromModel(
                    estimator=LogisticRegression(
                        penalty="l1", solver="liblinear", max_iter=self.max_iter
                    ),
                    max_features=self.max_features,
                    threshold=-np.inf,
                ),
            ),
        ]
        super().__init__(steps=steps, **kwargs)


class Classifier(Pipeline):
    def __init__(
        self,
        base_features,
        time_domain_groups: dict,
        freq_domain_groups: dict,
        max_extra_feat: int,
        max_iter: int,
        seed: int,
        **kwargs: Any,
    ) -> None:
        self.base_features = base_features
        self.time_domain_groups = time_domain_groups
        self.freq_domain_groups = freq_domain_groups
        self.max_extra_feat = max_extra_feat
        self.max_iter = max_iter
        self.seed = seed
        base_feat = [
            (
                "base_feat",
                ColumnTransformer(
                    [("selector", "passthrough", self.base_features)], remainder="drop"
                ),
            )
        ]
        time_domain_extra_feat = [
            (block, FeatureSelector(block, cols, self.max_iter, self.max_extra_feat))
            for block, cols in self.time_domain_groups.items()
        ]
        freq_domain_extra_feat = [
            (block, FeatureSelector(block, cols, self.max_iter, self.max_extra_feat))
            for block, cols in self.freq_domain_groups.items()
        ]
        steps = [
            (
                "features",
                FeatureUnion(
                    base_feat + time_domain_extra_feat + freq_domain_extra_feat
                ),
            ),
            ("classifier", RandomForestClassifier(n_jobs=-1, random_state=seed)),
        ]
        super().__init__(steps=steps, **kwargs)
