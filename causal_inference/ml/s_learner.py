__all__ = ["SLearner"]

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from causal_inference.ml.evaluation import CumulativeGainEvaluator
from causal_inference.ml._utils import _check_input_validity
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class SLearner(object):
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: Union[str, List[str]],
        max_depth: int = 3,
        min_child_samples: int = 30,
        test_size: float = 0.30,
        seed: int = 99,
    ) -> None:
        self._data = data
        self._outcome = outcome
        self._treatment = treatment
        self._covariates = _check_input_validity(
            covariates, exception=InvalidDataFormatForInputs
        )
        self._max_depth = max_depth
        self._min_child_samples = min_child_samples
        self._test_size = test_size
        self._seed = seed

        self._model = None

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            self._data, test_size=self._test_size, random_state=self._seed
        )

    def fit(self) -> None:
        np.random.seed(self._seed)
        self._model = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        train, _ = self._split_data()

        self._model.fit(
            train[self._covariates + [self._treatment]], train[self._outcome]
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        return self._model.predict(
            df[self._covariates].assign(**{self._treatment: 1.0})
        ) - self._model.predict(df[self._covariates].assign(**{self._treatment: 0.0}))

    def predict_train(self) -> pd.DataFrame:
        train, _ = self._split_data()
        return self.predict(train)

    def predict_test(self) -> pd.DataFrame:
        _, test = self._split_data()
        return self.predict(test)

    def eval(self) -> None:
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        train, test = self._split_data()
        evaluator = CumulativeGainEvaluator(
            train=train,
            test=test,
            model=self,
            outcome=self._outcome,
            treatment=self._treatment,
            covariates=self._covariates,
            model_name="S-Learner",
        )
        evaluator.fit()
        evaluator.eval()
