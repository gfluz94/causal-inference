__all__ = ["MatchingEstimator"]

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from joblib import Parallel, cpu_count, delayed
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from causal_inference.linear._utils import _check_input_validity
from causal_inference._exceptions.matching import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
    NoCovariatesAvailable,
)


class PropensityScoreModel(object):
    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        regularization: float = 1.0,
        categorical_covariates: Optional[Union[str, List[str]]] = None,
        numerical_covariates: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self._data = data
        self._treatment = treatment
        self._regularization = regularization
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
        if self._numerical_covariates:
            return StandardScaler()
        return None

    def _get_ohe(self) -> Optional[OneHotEncoder]:
        if self._categorical_covariates:
            return OneHotEncoder(drop="first", sparse_output=False)
        return None

    def _get_preprocessor(self) -> ColumnTransformer:
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
        if self._pipeline is None:
            return Pipeline(
                steps=[
                    self._get_preprocessor(),
                    LogisticRegression(C=self._regularization, random_state=99),
                ]
            )
        return self._pipeline

    def fit(self) -> None:
        self._pipeline = self._get_pipeline()
        self._pipeline = self._pipeline.fit(
            self._data[self._covariates], self._data[self._treatment]
        )

    def get_propensity_scores(self) -> np.ndarray:
        if self._pipeline is None:
            raise ModelNotFittedYet("Propensity Score Model needs to be fitted first!")
        return self._pipeline.predict_proba(self._data[self._covariates])[:, 1]

    def plot_distributions(self) -> None:
        score_column = "ps"
        df = self._data.assign({score_column: self.get_propensity_scores()})

        style.use("fivethirtyeight")
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.histplot(
            df.loc[df[self._treatment] == 1, score_column],
            color="green",
            alpha=0.3,
            bins=20,
            label="Treated",
            fill=True,
            ax=ax,
        )
        sns.histplot(
            df.loc[df[self._treatment] == 0, score_column],
            color="red",
            alpha=0.3,
            bins=20,
            label="Untreated",
            fill=True,
            ax=ax,
        )
        plt.xlabel("Propensity Score")
        plt.legend()
        plt.show()


class MatchingEstimator(object):
    _ATE = "ATE"
    _CI = "CI"
    _PROPENSITY_SCORE = "ps"

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        categorical_covariates: Optional[Union[str, List[str]]] = None,
        numerical_covariates: Optional[Union[str, List[str]]] = None,
        logistic_regression_regularization: float = 1.0,
    ) -> None:
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

        self._propensity_score_model = PropensityScoreModel(
            data=self._data,
            treatment=self._treatment,
            regularization=self._logistic_regression_regularization,
            categorical_covariates=self._categorical_covariates,
            numerical_covariates=self._numerical_covariates,
        )
        self._fitted = False
        self._results = None
        self._bootstrapped_ates = []

    def fit(self, check_distributions: bool = False) -> None:
        self._propensity_score_model.fit()
        if check_distributions:
            self._propensity_score_model.plot_distributions()
        self._fitted = True

    def _estimate_ate_for_sample(self, data: pd.DataFrame) -> float:
        if not self._fitted:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        return np.mean(
            data[self._outcome]
            * (data[self._treatment] - data[self._PROPENSITY_SCORE])
            / (data[self._PROPENSITY_SCORE] * (1 - data[self._PROPENSITY_SCORE]))
        )

    def _estimate_ate_with_bootstrapping(self) -> np.ndarray:
        np.random.seed(99)
        ates = Parallel(n_jobs=cpu_count() - 1)(
            delayed(self._estimate_ate_for_sample)(
                self._data.sample(frac=1.0, seed=seed)
            )
            for seed in range(1_000)
        )
        return np.array(ates)

    def _get_results(self) -> Dict[str, Any]:
        if not self._fitted:
            raise ModelNotFittedYet("Model needs to be fitted first!")

        if self._results is None and not self._bootstrapped_ates:
            self._data = self._data.assign(
                {
                    self._PROPENSITY_SCORE: self._propensity_score_model.get_propensity_scores()
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