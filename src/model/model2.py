from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import Any


class Classifier(Pipeline):
    def __init__(self, columns, **kwargs: Any) -> None:
        self.columns = columns
        steps = [("selector", ColumnTransformer([("selector", "passthrough", self.columns)], remainder="drop")),
                 ("classifier", RandomForestClassifier(n_jobs=-1))]
        super().__init__(steps=steps, **kwargs)
