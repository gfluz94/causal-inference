__all__ = ["OLSEstimator"]

from typing import Any, Dict, List, Optional, Tuple, Union
from joblib import delayed, Parallel, cpu_count
import pandas as pd
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from causal_inference.linear._utils import _check_input_validity
from causal_inference._exceptions.ols import (
    InvalidDataFormatForCovariates,
    ModelNotFittedYet,
    ATECannotBeEstimatedForHeterogeneousCase,
    CATECannotBeEstimatedForHomogeneousCase,
)


class OLSEstimator(object):
    """Class that encapsulates the whole logic for estimating ATE and CATE with linear models.

    Parameters:
        data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
        outcome (str): Name of column containing outcome data.
        treatment (str): Name of column containing treatment data.
        covariates (Union[str, List[str]], optional): Name(s) of column(s) containing covariates data. Defaults to None.
        heterogeneous (bool, optional): Whether to compute ATE (False) or CATE (True). Defaults to False.

    Raises:
        InvalidDataFormatForCovariates: Exception raised when `covariates` are neither List[str] or str.
        ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        ATECannotBeEstimatedForHeterogeneousCase: Exception raised when user attempts to estimate ATE for heterogeneous setting.
        CATECannotBeEstimatedForHomogeneousCase: Exception raised when user attempts to estimate CATE for homogeneous setting.

    """

    _ATE = "ATE"
    _CATE = "CATE"
    _SE = "SE"
    _CI = "CI"
    _PVALUE = "p-VALUE"

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: Optional[Union[str, List[str]]] = None,
        heterogeneous: bool = False,
    ) -> None:
        """Constructor method for OLSEstimator.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
            outcome (str): Name of column containing outcome data.
            treatment (str): Name of column containing treatment data.
            covariates (Union[str, List[str]], optional): Name(s) of column(s) containing covariates data. Defaults to None.
            heterogeneous (bool, optional): Whether to compute ATE (False) or CATE (True). Defaults to False.

        Raises:
            InvalidDataFormatForCovariates: Exception raised when `covariates` are neither List[str] or str
        """
        self._data = data
        self._outcome = outcome
        self._treatment = treatment
        self._covariates = []
        self._heterogeneous = heterogeneous
        self._covariates = _check_input_validity(
            covariates, InvalidDataFormatForCovariates
        )

        self._model = None
        self._results = None
        self._coeffs_interaction = None

    @property
    def raw_data(self) -> pd.DataFrame:
        """(pd.DataFrame) pandas dataframe containing treatment, outcome and covariates"""
        return self._data

    @property
    def outcome(self) -> str:
        """(str) Name of column containing outcome data"""
        return self._outcome

    @property
    def treatment(self) -> str:
        """(str) Name of column containing treatment data"""
        return self._treatment

    @property
    def covariates(self) -> Union[str, List[str]]:
        """(Union[str, List[str]]) Name(s) of column(s) containing covariates data"""
        return self._covariates

    @property
    def heterogeneous(self) -> bool:
        """(bool) Whether to compute ATE (False) or CATE (True)"""
        return self._heterogeneous

    def fit(self) -> None:
        """Method that fits the estimator on input dasta."""
        X = self._treatment
        if self._covariates:
            if self._heterogeneous:
                X = " + ".join(
                    f"{t}*{x}"
                    for t, x in zip(
                        [self._treatment] * len(self._covariates), self._covariates
                    )
                )
            else:
                X = f"{self._treatment} + " + " + ".join(self._covariates)
        self._model = smf.ols(formula=f"{self._outcome} ~ {X}", data=self._data).fit()
        if self._heterogeneous:
            self._get_cis_for_heterogeneity()
            self.estimate_cate = self._estimate_cate
        else:
            self._get_results()
            self.estimate_ate = self._estimate_ate

    def _get_results(self) -> None:
        """Method that computes ATE along with confidence intervals for the estimates.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        ate = self._model.params[self._treatment]
        se = self._model.bse[self._treatment]
        lower_ci, upper_ci = self._model.conf_int().loc[self._treatment].values
        p_value = self._model.pvalues[self._treatment]
        self._results = {
            self._ATE: np.round(ate, 5),
            self._SE: np.round(se, 5),
            self._CI: (np.round(lower_ci, 5), np.round(upper_ci)),
            self._PVALUE: np.round(p_value, 5),
        }

    def _get_cis_for_heterogeneity(self) -> None:
        """Method that computes CATE along with confidence intervals for the estimates.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        self._coeffs_interaction = {}
        preffix = f"{self._treatment}:"
        for coeff_name in self._model.conf_int().index:
            if coeff_name.startswith(preffix):
                lower, upper = self._model.conf_int().loc[coeff_name]
                self._coeffs_interaction[coeff_name.replace(preffix, "")] = (
                    np.round(lower, 5),
                    np.round(upper, 5),
                )
        lower, upper = self._model.conf_int().loc[self._treatment]
        self._coeffs_interaction[self._treatment] = np.round(lower, 5), np.round(
            upper, 5
        )

    def _estimate_ate(self, plot_result: bool = False) -> Dict[str, Any]:
        """Method that returns ATE results along with the plot, optionally.

        Args:
            plot_result (bool, optional): Whether or not to display plot with results. Defaults to False.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
            ATECannotBeEstimatedForHeterogeneousCase: Exception raised when user attempts to estimate ATE for heterogeneous setting.

        Returns:
            Dict[str, Any]: Dictionary {variable: value} containing effect, standard error, confidence interval and p-value.
        """
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        if self._heterogeneous:
            raise ATECannotBeEstimatedForHeterogeneousCase(
                "`heterogeneous` set to True, so computing ATE is not allowed!"
            )
        if plot_result:
            loc = self._results[self._ATE]
            scale = self._results[self._SE]
            ci = self._results[self._CI]
            data = np.random.normal(loc=loc, scale=scale, size=10_000)
            self._plot_result(data=data, effect=loc, ci=ci)
        return self._results

    def _estimate_cate_from_sample(
        self, covariates: Dict[str, Any], random_state: int
    ) -> float:
        """Method that returns CATE estimate for a sample.

        Args:
            covariates (Dict[str, Any]): Dictionary {covariate: value} for the individual for which we want to estimate CATE.
            random_state (int): Seed to ensure reproducibility for the bootstrapping process.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
            CATECannotBeEstimatedForHomogeneousCase: Exception raised when user attempts to estimate CATE for homogeneous setting.

        Returns:
            float: Single estimate for CATE.
        """
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        if not self._heterogeneous:
            raise CATECannotBeEstimatedForHomogeneousCase(
                "`heterogeneous` set to False, so computing CATE is not allowed!"
            )
        covariates_values = covariates.copy()
        covariates_values[self._treatment] = 1
        np.random.seed(random_state)
        cate = sum(
            np.round(np.random.uniform(*ci) * covariates_values[coeff_name], 5)
            for coeff_name, ci in self._coeffs_interaction.items()
        )
        return cate

    def _estimate_cate_bootstrap(self, covariates: Dict[str, Any]) -> np.ndarray:
        """Method that returns CATE estimates with bootstrapping.

        Args:
            covariates (Dict[str, Any]): Dictionary {covariate: value} for the individual for which we want to estimate CATE.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
            CATECannotBeEstimatedForHomogeneousCase: Exception raised when user attempts to estimate CATE for homogeneous setting.

        Returns:
            np.ndarray: Array containing 10,000 estimates for CATE with bootstrapping.
        """
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")
        if not self._heterogeneous:
            raise CATECannotBeEstimatedForHomogeneousCase(
                "`heterogeneous` set to False, so computing CATE is not allowed!"
            )
        n_jobs = cpu_count() - 1
        cates = Parallel(n_jobs=n_jobs)(
            delayed(self._estimate_cate_from_sample)(covariates, seed)
            for seed in range(10_000)
        )
        return np.array(cates)

    def _estimate_cate(
        self, covariates: Dict[str, Any], plot_result: bool = False
    ) -> Dict[str, Any]:
        """Method that returns CATE results along with the plot, optionally.

        Args:
            covariates (Dict[str, Any]): Dictionary {covariate: value} for the individual for which we want to estimate CATE.
            plot_result (bool, optional): Whether or not to display plot with results. Defaults to False.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
            CATECannotBeEstimatedForHomogeneousCase: Exception raised when user attempts to estimate CATE for homogeneous setting.

        Returns:
            Dict[str, Any]: Dictionary {variable: value} containing effect and confidence interval estimated with bootstrapping.
        """
        cates = self._estimate_cate_bootstrap(covariates=covariates)
        cate = np.round(np.mean(cates), 5)
        cate_ci = (
            np.round(np.quantile(cates, q=0.025), 5),
            np.round(np.quantile(cates, q=0.975), 5),
        )
        if plot_result:
            self._plot_result(data=cates, effect=cate, ci=cate_ci)
        return {
            self._CATE: cate,
            self._CI: cate_ci,
        }

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
