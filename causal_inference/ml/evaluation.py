__all__ = ["CumulativeGainEvaluator"]

from typing import List
import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt

from causal_inference.ml.base import MetaLearner
from causal_inference._exceptions.ml import EvaluatorNotFittedYet


class CumulativeGainEvaluator(object):
    """Class that encapsulates whole logic for evaluating Meta-Learners, based on Cumulative Gain Curve.

    Parameters:
        train (pd.DataFrame): Training set, containing outcome, covariates, and treatment variables.
        test (pd.DataFrame): Test set, containing outcome, covariates, and treatment variables.
        model (MetaLearner): Meta-Learner class under `ml` module
        outcome (str): Name of column containing outcome data.
        treatment (str): Name of column containing treatment data.
        covariates (List[str]): Name(s) of column(s) containing covariates data.
        model_name (str): Name of the meta-learner to be displayed on the cumulative gain curve.

    Raises:
        EvaluatorNotFittedYet: Exception raised when results are requested, but evaluator has not been fitted yet.
    """

    _CUMULATIVE_ELASTICITY = "cumulative_elasticity"
    _CUMULATIVE_GAIN = "cumulative_gain"
    _CATE = "cate"
    _X_LABEL = "sample_frac"

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model: MetaLearner,
        outcome: str,
        treatment: str,
        covariates: List[str],
        model_name: str,
    ) -> None:
        """Constructor method for CumulativeGainEvaluator.

        Args:
            train (pd.DataFrame): Training set, containing outcome, covariates, and treatment variables.
            test (pd.DataFrame): Test set, containing outcome, covariates, and treatment variables.
            model (MetaLearner): Meta-Learner class under `ml` module
            outcome (str): Name of column containing outcome data.
            treatment (str): Name of column containing treatment data.
            covariates (List[str]): Name(s) of column(s) containing covariates data.
            model_name (str): Name of the meta-learner to be displayed on the cumulative gain curve.

        Raises:
            EvaluatorNotFittedYet: Exception raised when results are requested, but evaluator has not been fitted yet.
        """
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
        """(pd.DataFrame) Training set, containing outcome, covariates, and treatment variables"""
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        """(pd.DataFrame) Test set, containing outcome, covariates, and treatment variables"""
        return self._test

    def _compute_cate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method that computes CATE, on top of meta-learner `predict()` method.

        Args:
            df (pd.DataFrame): Dataframe containing covariates, and treatment variables.

        Returns:
            pd.DataFrame: Dataframe containing inputs and estimated CATE.
        """
        return df.assign(**{self._CATE: self._model.predict(df)})

    def _sort_by_cate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method that sorts dataframe by CATE, in a decreasing fashion.

        Args:
            df (pd.DataFrame): Dataframe containing covariates, and treatment variables.

        Returns:
            pd.DataFrame: Sorted dataframe by CATE, in a decreasing fashion.
        """
        return df.sort_values(by=self._CATE, ascending=False)

    def _compute_cumulative_elasticity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method that computes cumulative elasticity on top of CATE values.

        Args:
            df (pd.DataFrame): Dataframe containing estimated CATE, covariates, and treatment variables.

        Returns:
            pd.DataFrame: Input dataframe with computed cumulative elasticity for each entry.
        """
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
        """Method that computes cumulative gain for further plotting of Cumulative Gain Curve.

        Args:
            df (pd.DataFrame): Dataframe containing estimated CATE, covariates, and treatment variables.

        Returns:
            pd.DataFrame: Input dataframe with cumulative gain computed for each entry.
        """
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
        """Method that fits the evaluator - ie, it computes cumulative gain for training and test sets."""
        self._train = self._compute_cumulative_gain(self._train)
        self._test = self._compute_cumulative_gain(self._test)
        self._fitted = True

    def eval(self) -> None:
        """Method that plots the Cumulative Gain Curve, after cumulative gains have already been computed.
        It is similar to a ROC-Curve: the higher the area under the curve, the better the meta-learner's performance.

        Raises:
            EvaluatorNotFittedYet: Exception raised when results are requested, but evaluator has not been fitted yet.
        """
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
