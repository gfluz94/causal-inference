__all__ = ["MatchingEstimator"]

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
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
    pass
