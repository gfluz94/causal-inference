__all__ = ["XLearner"]

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor

from causal_inference.ml.base import MetaLearner
from causal_inference.ml.evaluation import CumulativeGainEvaluator
from causal_inference.linear._utils import _check_input_validity
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class XLearner(MetaLearner):
    """Class for X-Learner. It estimates CATE by fitting four separate models, according to treatment, and using matching.

    Parameters:
        data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
        outcome (str): Name of column containing outcome data.
        treatment (str): Name of column containing treatment data.
        covariates_categorical (Union[str, List[str]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
        covariates_numerical (Union[str, List[str]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.
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
        covariates_categorical: Optional[Union[str, List[str]]] = None,
        covariates_numerical: Optional[Union[str, List[str]]] = None,
        max_depth: int = 3,
        min_child_samples: Optional[int] = None,
        test_size: float = 0.30,
        seed: int = 99,
    ) -> None:
        """Constructor method for XLearner.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
            outcome (str): Name of column containing outcome data.
            treatment (str): Name of column containing treatment data.
            covariates_categorical (Union[str, List[str]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
            covariates_numerical (Union[str, List[str]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.
            max_depth (int, optional): Maximum depth of LGBM Regressor trees. Defaults to 3.
            min_child_samples (int, optional): Minimum childs to split further in maximum depth of LGBM Regressor trees. Defaults to None.
            test_size (float, optional): Test size for in-sample hold-out validation. Defaults to 0.30.
            seed (int, optional): Random seed for reproducibility. Defaults to 99.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
            InvalidDataFormatForInputs: Exception raised when inputs are neither List[str] or str.
        """
        super(XLearner, self).__init__(data=data, test_size=test_size, seed=seed)
        self._outcome = outcome
        self._treatment = treatment
        covariates_categorical = _check_input_validity(
            covariates_categorical, exception=InvalidDataFormatForInputs
        )
        covariates_numerical = _check_input_validity(
            covariates_numerical, exception=InvalidDataFormatForInputs
        )
        self._covariates_categorical = (
            covariates_categorical if covariates_categorical is not None else []
        )
        self._covariates_numerical = (
            covariates_numerical if covariates_numerical is not None else []
        )
        self._covariates = covariates_numerical + covariates_categorical
        self._max_depth = max_depth
        self._min_child_samples = min_child_samples

        self._model_T0 = None
        self._model_T1 = None
        self._model_tau0 = None
        self._model_tau1 = None
        self._model_ps = None

    def fit(self) -> None:
        """Method that fits the estimator on the training set of the input data."""
        # Models
        np.random.seed(self._seed)
        self._model_T0 = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        self._model_T1 = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        self._model_tau0 = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        self._model_tau1 = LGBMRegressor(
            max_depth=self._max_depth,
            min_child_samples=self._min_child_samples,
            random_state=self._seed,
        )
        self._model_ps = Pipeline(
            steps=[
                (
                    "preprocessing",
                    ColumnTransformer(
                        transformers=[
                            ("scale", StandardScaler(), self._covariates_numerical),
                            (
                                "ohe",
                                OneHotEncoder(drop="first", sparse_output=False),
                                self._covariates_categorical,
                            ),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("lr", LogisticRegression(random_state=self._seed)),
            ]
        )

        # Training data
        train, _ = self._split_data()

        # 1st Stage
        self._model_T0.fit(
            train.query(f"{self._treatment} == 0")[self._covariates],
            train.query(f"{self._treatment} == 0")[self._outcome],
        )
        self._model_T1.fit(
            train.query(f"{self._treatment} == 1")[self._covariates],
            train.query(f"{self._treatment} == 1")[self._outcome],
        )

        # 2nd Stage
        tau = np.where(
            train[self._treatment] == 0,
            self._model_T1.predict(train[self._covariates]) - train[self._outcome],
            train[self._outcome] - self._model_T0.predict(train[self._covariates]),
        )

        # 3rd Stage
        self._model_tau0.fit(
            train.query(f"{self._treatment} == 0")[self._covariates],
            tau[train[self._treatment] == 0],
        )
        self._model_tau1.fit(
            train.query(f"{self._treatment} == 1")[self._covariates],
            tau[train[self._treatment] == 1],
        )

        # 4th Stage
        self._model_ps.fit(train[self._covariates], train[self._treatment])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Method that predicts CATE for an input dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe containing covariates and treatment.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            np.ndarray: Numpy array containing CATE predictions.
        """
        if (
            self._model_T0 is None
            or self._model_T1 is None
            or self._model_tau0 is None
            or self._model_tau1 is None
            or self._model_ps is None
        ):
            raise ModelNotFittedYet("Model needs to be fitted first!")
        return self._model_ps.predict_proba(df[self._covariates])[
            :, 1
        ] * self._model_tau0.predict(df[self._covariates]) + (
            1 - self._model_ps.predict_proba(df[self._covariates])[:, 1]
        ) * self._model_tau1.predict(
            df[self._covariates]
        )

    def eval(self) -> None:
        """Method that runs evaluation with Cumulative Gain Curve.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        if (
            self._model_T0 is None
            or self._model_T1 is None
            or self._model_tau0 is None
            or self._model_tau1 is None
            or self._model_ps is None
        ):
            raise ModelNotFittedYet("Model needs to be fitted first!")
        train, test = self._split_data()
        evaluator = CumulativeGainEvaluator(
            train=train,
            test=test,
            model=self,
            outcome=self._outcome,
            treatment=self._treatment,
            covariates=self._covariates,
            model_name="X-Learner",
        )
        evaluator.fit()
        evaluator.eval()
