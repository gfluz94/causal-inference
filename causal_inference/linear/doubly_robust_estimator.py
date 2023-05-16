__all__ = ["DoublyRobustEstimator"]

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from joblib import Parallel, cpu_count, delayed
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from causal_inference.linear._utils import _check_input_validity
from causal_inference.linear.matching import PropensityScoreModel
from causal_inference._exceptions.doubly_robust_estimator import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
    NoCovariatesAvailable,
)


class LinearRegressionModel(object):
    """Class that encapsulates the whole logic for estimating µ0 and µ1 for Doubly Robust Estimation.

    Parameters:
        data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
        outcome (str): Name of column containing outcome data.
        categorical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
        numerical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.

    Raises:
        NoCovariatesAvailable: Exception raised when no covariates are passed as arguments.
        ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        categorical_covariates: Optional[Union[str, List[str]]] = None,
        numerical_covariates: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Constructor method for LinearRegressionModel.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
            outcome (str): Name of column containing outcome data.
            categorical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
            numerical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.

        Raises:
            NoCovariatesAvailable: Exception raised when no covariates are passed as arguments.
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        self._data = data
        self._outcome = outcome
        self._categorical_covariates = _check_input_validity(
            categorical_covariates, InvalidDataFormatForInputs
        )
        self._numerical_covariates = _check_input_validity(
            numerical_covariates, InvalidDataFormatForInputs
        )
        self._covariates = (
            self._categorical_covariates if self._categorical_covariates else []
        ) + (self._numerical_covariates if self._numerical_covariates else [])

        if not self._categorical_covariates and not self._numerical_covariates:
            raise NoCovariatesAvailable(
                "At least one covariate needs to be informed for Propensity Score Matching."
            )

        self._pipeline = None

    def _get_scaler(self) -> Optional[StandardScaler]:
        """Method that returns Scaler object, if numerical covariates available.

        Returns:
            Optional[StandardScaler]: Standard Scaler for numerical covariates.
        """
        if self._numerical_covariates:
            return StandardScaler()
        return None

    def _get_ohe(self) -> Optional[OneHotEncoder]:
        """Method that returns Categorical Encoder object, if categorical covariates available.

        Returns:
            Optional[OneHotEncoder]: One-Hot Encoder for categorical covariates.
        """
        if self._categorical_covariates:
            return OneHotEncoder(drop="first", sparse_output=False)
        return None

    def _get_preprocessor(self) -> ColumnTransformer:
        """Method that returns the whole preprocessing pipeline.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        scaler = self._get_scaler()
        ohe = self._get_ohe()

        preprocessing_steps = []
        if scaler is not None:
            preprocessing_steps.append(("scaler", scaler, self._numerical_covariates))
        if ohe is not None:
            preprocessing_steps.append(("ohe", ohe, self._categorical_covariates))

        return ColumnTransformer(
            transformers=preprocessing_steps, remainder="passthrough"
        )

    def _get_pipeline(self) -> Pipeline:
        """Method that returns the whole pipeline: preprocessing + logistic regression.

        Returns:
            Pipeline: Whole model pipeline (preprocessing + logistic regression).
        """
        if self._pipeline is None:
            return Pipeline(
                steps=[
                    ("preprocessor", self._get_preprocessor()),
                    ("lr", LinearRegression()),
                ]
            )
        return self._pipeline

    def fit(self) -> None:
        """Method that fits the estimator on input data."""
        self._pipeline = self._get_pipeline()
        self._pipeline = self._pipeline.fit(
            self._data[self._covariates], self._data[self._outcome]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Method that returns the expected values for treated or control groups.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            np.ndarray: Array of propensity scores.
        """
        if self._pipeline is None:
            raise ModelNotFittedYet("Propensity Score Model needs to be fitted first!")
        return self._pipeline.predict(X)


class DoublyRobustEstimator(object):
    """Class that encapsulates the whole logic for estimating ATE with Doubly Robust Estimation.

    Parameters:
        data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
        outcome (str): Name of column containing outcome data.
        treatment (str): Name of column containing treatment data.
        categorical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
        numerical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.
        logistic_regression_regularization (float, optional): Inverse of regularization strength. Defaults to 1.0.

    Raises:
        ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

    """

    _ATE = "ATE"
    _CI = "CI"
    _PROPENSITY_SCORE = "ps"
    _MU_0 = "mu0"
    _MU_1 = "mu0"

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        categorical_covariates: Optional[Union[str, List[str]]] = None,
        numerical_covariates: Optional[Union[str, List[str]]] = None,
        logistic_regression_regularization: float = 1.0,
    ) -> None:
        """Constructor method for DoublyRobustEstimator.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
            outcome (str): Name of column containing outcome data.
            treatment (str): Name of column containing treatment data.
            categorical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
            numerical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.
            logistic_regression_regularization (float, optional): Inverse of regularization strength. Defaults to 1.0.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        self._data = data
        self._outcome = outcome
        self._treatment = treatment
        self._categorical_covariates = _check_input_validity(
            categorical_covariates, InvalidDataFormatForInputs
        )
        self._numerical_covariates = _check_input_validity(
            numerical_covariates, InvalidDataFormatForInputs
        )
        self._logistic_regression_regularization = logistic_regression_regularization

        self._covariates = (
            self._categorical_covariates if self._categorical_covariates else []
        ) + (self._numerical_covariates if self._numerical_covariates else [])

        self._propensity_score_model = PropensityScoreModel(
            data=self._data,
            treatment=self._treatment,
            regularization=self._logistic_regression_regularization,
            categorical_covariates=self._categorical_covariates,
            numerical_covariates=self._numerical_covariates,
        )
        self._mu0_model = LinearRegressionModel(
            data=self._data.query(f"{self._treatment} == 0"),
            outcome=self._outcome,
            categorical_covariates=self._categorical_covariates,
            numerical_covariates=self._numerical_covariates,
        )
        self._mu1_model = LinearRegressionModel(
            data=self._data.query(f"{self._treatment} == 1"),
            outcome=self._outcome,
            categorical_covariates=self._categorical_covariates,
            numerical_covariates=self._numerical_covariates,
        )
        self._fitted = False
        self._results = None
        self._bootstrapped_ates = []

    def fit(self, check_distributions: bool = False) -> None:
        """Method that fits the estimator on input data.

        Args:
            check_distributions (bool, optional): Whether or not to plot propensity scores distributions. Defaults to False.
        """
        self._propensity_score_model.fit()
        self._mu0_model.fit()
        self._mu1_model.fit()
        if check_distributions:
            self._propensity_score_model.plot_distributions()
        self._fitted = True

    def _estimate_ate_for_sample(self, data: pd.DataFrame) -> float:
        """Method that returns ATE results for a sampled dataset.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            float: Estimated ATE value.
        """
        if not self._fitted:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        return np.mean(
            data[self._treatment]
            * (data[self._outcome] - data[self._MU_1])
            / data[self._PROPENSITY_SCORE]
            + data[self._MU_1]
        ) - np.mean(
            (1 - data[self._treatment])
            * (data[self._outcome] - data[self._MU_0])
            / (1 - data[self._PROPENSITY_SCORE])
            + data[self._MU_0]
        )

    def _estimate_ate_with_bootstrapping(self) -> np.ndarray:
        """Method that estimates ATE 1,000 times using bootstrapping.
        The main goal is to compute the confidence interval for ATE final estimate.

        Returns:
            np.ndarray: Array containing estimates of ATE.
        """
        np.random.seed(99)
        ates = Parallel(n_jobs=cpu_count() - 1)(
            delayed(self._estimate_ate_for_sample)(
                self._data.sample(frac=1.0, replace=True, random_state=seed)
            )
            for seed in range(1_000)
        )
        return np.array(ates)

    def _get_results(self) -> Dict[str, Any]:
        """Method that computes ATE along with confidence intervals for the estimates.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        if not self._fitted:
            raise ModelNotFittedYet("Model needs to be fitted first!")

        if self._results is None and not self._bootstrapped_ates:
            self._data = self._data.assign(
                **{
                    self._PROPENSITY_SCORE: self._propensity_score_model.get_propensity_scores(),
                    self._MU_0: self._mu0_model.predict(self._data[self._covariates]),
                    self._MU_1: self._mu1_model.predict(self._data[self._covariates]),
                }
            )
            ate = self._estimate_ate_for_sample(self._data)
            self._bootstrapped_ates = self._estimate_ate_with_bootstrapping()
            lower, upper = np.percentile(self._bootstrapped_ates, 2.5), np.percentile(
                self._bootstrapped_ates, 97.5
            )

            self._results = {
                self._ATE: np.round(ate, 5),
                self._CI: (np.round(lower, 5), np.round(upper, 5)),
            }

        return self._results

    def estimate_ate(self, plot_result: bool = False) -> Dict[str, Any]:
        """Method that returns ATE results along with the plot, optionally.

        Args:
            plot_result (bool, optional): Whether or not to display plot with results. Defaults to False.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            Dict[str, Any]: Dictionary {variable: value} containing effect, standard error, confidence interval and p-value.
        """
        output = self._get_results()
        if plot_result:
            self._plot_result(
                data=self._bootstrapped_ates,
                effect=output[self._ATE],
                ci=output[self._CI],
            )
        return output

    def _plot_result(
        self, data: np.array, effect: float, ci: Tuple[float, float]
    ) -> None:
        """Method that returns a plot with final effect estimate along with confidence interval.

        Args:
            data (np.array): Data points of estimates to plot distribution.
            effect (float): Actual effect computed (mean).
            ci (Tuple[float, float]): Confidence interval for the effect estimate.
        """
        style.use("fivethirtyeight")
        lower, upper = ci
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(x=data, bins=30)
        plt.vlines(
            x=lower, ymin=0, ymax=ax.dataLim.max[-1], linestyles=["--"], color="red"
        )
        plt.vlines(
            x=upper, ymin=0, ymax=ax.dataLim.max[-1], linestyles=["--"], color="red"
        )
        plt.title(
            f"Estimated Effect = {effect:.2f} [95% CI ({lower:.2f}, {upper:.2f})]",
            fontsize=13,
        )
        plt.show()
