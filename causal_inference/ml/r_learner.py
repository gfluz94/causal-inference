__all__ = ["RLearner"]

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from lightgbm import LGBMRegressor

from causal_inference.ml.evaluation import CumulativeGainEvaluator
from causal_inference.ml._utils import _check_input_validity
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class RLearner(object):
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: Union[str, List[str]],
        max_depth: int = 3,
        min_child_samples: Optional[int] = None,
        test_size: float = 0.30,
        cv: int = 3,
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
        self._cv = cv
        self._seed = seed

        self._model_Yres = None
        self._model_Tres = None
        self._model = None

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            self._data, test_size=self._test_size, random_state=self._seed
        )

    def fit(self) -> None:
        np.random.seed(self._seed)
        self._model_Yres = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        self._model_Tres = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        self._model = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        train, _ = self._split_data()

        train = train.assign(
            **{
                f"{self._outcome}_res": train[self._outcome]
                - cross_val_predict(
                    self._model_Yres,
                    train[self._covariates],
                    train[self._outcome],
                    cv=self._cv,
                ),
                f"{self._treatment}_res": train[self._treatment]
                - cross_val_predict(
                    self._model_Tres,
                    train[self._covariates],
                    train[self._treatment],
                    cv=self._cv,
                ),
            }
        )

        y_adj = train[f"{self._outcome}_res"] / train[f"{self._treatment}_res"]
        w = train[f"{self._treatment}_res"] ** 2

        self._model.fit(train[self._covariates], y_adj, sample_weight=w)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._model_Yres is None or self._model_Tres is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        return self._model.predict(df[self._covariates])

    def predict_train(self) -> pd.DataFrame:
        train, _ = self._split_data()
        return self.predict(train)

    def predict_test(self) -> pd.DataFrame:
        _, test = self._split_data()
        return self.predict(test)

    def eval(self) -> None:
        if self._model is None or self._model_Yres is None or self._model_Tres is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        train, test = self._split_data()
        evaluator = CumulativeGainEvaluator(
            train=train,
            test=test,
            model=self,
            outcome=self._outcome,
            treatment=self._treatment,
            covariates=self._covariates,
            model_name="R-Learner",
        )
        evaluator.fit()
        evaluator.eval()
