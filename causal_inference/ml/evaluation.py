__all__ = ["CumulativeGainEvaluator"]

from typing import List
import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

from causal_inference._exceptions.ml import EvaluatorNotFittedYet


class CumulativeGainEvaluator(object):
    _CUMULATIVE_ELASTICITY = "cumulative_elasticity"
    _CUMULATIVE_GAIN = "cumulative_gain"
    _CATE = "cate"
    _X_LABEL = "sample_frac"

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model: BaseEstimator,
        outcome: str,
        treatment: str,
        covariates: List[str],
        model_name: str,
    ) -> None:
        self._train = train.copy()
        self._test = test.copy()
        self._model = model
        self._outcome = outcome
        self._treatment = treatment
        self._covariates = covariates
        self._model_name = model_name

        self._fitted = False

    @property
    def train(self) -> pd.DataFrame:
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        return self._test

    def _compute_cate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{self._CATE: self._model.predict(df)})

    def _sort_by_cate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=self._CATE, ascending=False)

    def _compute_cumulative_elasticity(self, df: pd.DataFrame) -> pd.DataFrame:
        elasticity = [np.nan]
        for idx in range(2, len(df) + 1):
            cur_df = df.iloc[:idx]
            current_avg_treatment = cur_df[self._treatment].mean()
            current_avg_outcome = cur_df[self._outcome].mean()
            numerator = np.sum(
                (cur_df[self._treatment] - current_avg_treatment)
                * (cur_df[self._outcome] - current_avg_outcome)
            )
            denominator = np.sum((cur_df[self._treatment] - current_avg_treatment) ** 2)
            elasticity.append(numerator / denominator)
        return df.assign(**{self._CUMULATIVE_ELASTICITY: np.array(elasticity)})

    def _compute_cumulative_gain(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_cate(df)
        df = self._sort_by_cate(df)

        df = df.assign(**{self._X_LABEL: np.arange(1, len(df) + 1) / len(df)})
        df = self._compute_cumulative_elasticity(df)

        return df.assign(
            **{
                self._CUMULATIVE_GAIN: lambda x: x[self._CUMULATIVE_ELASTICITY]
                * x[self._X_LABEL]
            }
        )

    def fit(self) -> None:
        self._train = self._compute_cumulative_gain(self._train)
        self._test = self._compute_cumulative_gain(self._test)
        self._fitted = True

    def eval(self) -> None:
        if not self._fitted:
            raise EvaluatorNotFittedYet("Call `.fit()` before `.eval()`!")
        style.use("fivethirtyeight")
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(
            self._train[self._X_LABEL],
            self._train[self._CUMULATIVE_GAIN],
            label="train",
        )
        ax.plot(
            self._test[self._X_LABEL], self._test[self._CUMULATIVE_GAIN], label="test"
        )
        ax.plot(
            self._test.loc[
                ~self._test[self._CUMULATIVE_GAIN].isnull(), self._X_LABEL
            ].values[[0, -1]],
            self._test.loc[
                ~self._test[self._CUMULATIVE_GAIN].isnull(), self._CUMULATIVE_GAIN
            ].values[[0, -1]],
            linestyle="--",
            color="grey",
            linewidth=2,
            label="random",
        )
        ax.set_title(f"Cumulative Gain Curve [{self._model_name}]")
        ax.set_xlabel("% of Sample")
        ax.set_ylabel("Cumulative Gain")
        plt.legend()
        plt.show()
