__all__ = ["SLearner"]

from typing import List, Union
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from causal_inference.ml.base import MetaLearner
from causal_inference.ml.evaluation import CumulativeGainEvaluator
from causal_inference.ml._utils import _check_input_validity
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class SLearner(MetaLearner):
    """Class for S-Learner (aka Go-Horse Learner). It estimates CATE by artificially setting treatment levels and subtracting predictions.

    Parameters:
        data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
        outcome (str): Name of column containing outcome data.
        treatment (str): Name of column containing treatment data.
        covariates (Union[str, List[str]], optional): Name(s) of column(s) containing covariates data. Defaults to None.
        max_depth (int, optional): Maximum depth of LGBM Regressor trees. Defaults to 3.
        min_child_samples (int, optional): Minimum childs to split further in maximum depth of LGBM Regressor trees. Defaults to 30.
        test_size (float, optional): Test size for in-sample hold-out validation. Defaults to 0.30.
        seed (int, optional): Random seed for reproducibility. Defaults to 99.

    Raises:
        ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
    """

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
        """Constructor method for SLearner.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
            outcome (str): Name of column containing outcome data.
            treatment (str): Name of column containing treatment data.
            covariates (Union[str, List[str]], optional): Name(s) of column(s) containing covariates data. Defaults to None.
            max_depth (int, optional): Maximum depth of LGBM Regressor trees. Defaults to 3.
            min_child_samples (int, optional): Minimum childs to split further in maximum depth of LGBM Regressor trees. Defaults to 30.
            test_size (float, optional): Test size for in-sample hold-out validation. Defaults to 0.30.
            seed (int, optional): Random seed for reproducibility. Defaults to 99.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
            InvalidDataFormatForInputs: Exception raised when inputs are neither List[str] or str.
        """
        super(SLearner, self).__init__(data=data, test_size=test_size, seed=seed)
        self._outcome = outcome
        self._treatment = treatment
        self._covariates = _check_input_validity(
            covariates, exception=InvalidDataFormatForInputs
        )
        self._max_depth = max_depth
        self._min_child_samples = min_child_samples

        self._model = None

    def fit(self) -> None:
        """Method that fits the estimator on the training set of the input data."""
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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Method that predicts CATE for an input dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe containing covariates and treatment.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            np.ndarray: Numpy array containing CATE predictions.
        """
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        return self._model.predict(
            df[self._covariates].assign(**{self._treatment: 1.0})
        ) - self._model.predict(df[self._covariates].assign(**{self._treatment: 0.0}))

    def eval(self) -> None:
        """Method that runs evaluation with Cumulative Gain Curve.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
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
