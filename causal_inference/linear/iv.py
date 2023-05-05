__all__ = ["IVEstimator"]

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from linearmodels import IV2SLS

from causal_inference._exceptions.iv import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


def _check_input_validity(input_variable: Any) -> Optional[List[str]]:
    """Auxiliary function that checks whether the input belongs to one of the allowed types.

    Args:
        input_variable (Any): The input variable - a parameter passed to the class.

    Raises:
        InvalidDataFormatForInputs: Raised when `input_variable` is neither a list nor a string.

    Returns:
        Optional[List[str]]: List of str.
    """
    if isinstance(input_variable, str):
        return [input_variable]
    elif isinstance(input_variable, list) or input_variable is None:
        return input_variable
    else:
        raise InvalidDataFormatForInputs("Inputs must be either List[str] or str!")


class IVEstimator(object):
    """Class that encapsulates the whole logic for estimating ATE with instrumental variables.

    Parameters:
        data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
        outcome (str): Name of column containing outcome data.
        treatment (str): Name of column containing treatment data.
        instruments (Union[str, List[str]]): Name(s) of column(s) containing intrument(s) data.
        categorical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
        numerical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.

    Raises:
        ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

    """

    _ATE = "ATE"
    _SE = "SE"
    _CI = "CI"
    _PVALUE = "p-VALUE"

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        instruments: Union[str, List[str]],
        categorical_covariates: Optional[Union[str, List[str]]] = None,
        numerical_covariates: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Constructor method for IVEstimator.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
            outcome (str): Name of column containing outcome data.
            treatment (str): Name of column containing treatment data.
            instruments (Union[str, List[str]]): Name(s) of column(s) containing intrument(s) data.
            categorical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing categorical covariates data. Defaults to None.
            numerical_covariates (Optional[Union[str, List[str]]], optional): Name(s) of column(s) containing numerical covariates data. Defaults to None.
        """
        self._data = data
        self._outcome = outcome
        self._treatment = treatment
        self._instruments = _check_input_validity(instruments)
        self._categorical_covariates = _check_input_validity(categorical_covariates)
        self._numerical_covariates = _check_input_validity(numerical_covariates)

        self._model = None
        self._results = None

    def fit(self) -> None:
        """Method that fits the estimator on input data."""
        expression = ""

        # Writing formula for first stage
        Zs = " + ".join(self._instruments)
        first_stage = f"[{self._treatment} ~ {Zs}]"

        # Writing covariates expressions
        if self._categorical_covariates is not None:
            cat_covs = " + ".join(
                map(lambda x: f"C({x})", self._categorical_covariates)
            )
            expression += cat_covs + " + "

        if self._numerical_covariates is not None:
            num_covs = " + ".join(self._numerical_covariates)
            expression += num_covs + " + "

        self._model = IV2SLS.from_formula(
            formula=f"{self._outcome} ~ 1 + {expression} {first_stage}", data=self._data
        ).fit()

    def _get_results(self) -> Dict[str, Any]:
        """Method that computes ATE along with confidence intervals for the estimates.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        if self._model is None:
            raise ModelNotFittedYet("Model needs to be fitted first!")

        if self._results is None:
            ate = self._model.params[self._treatment]
            se = self._model.std_errors[self._treatment]
            lower, upper = self._model.conf_int().loc[self._treatment].values
            p_value = self._model.pvalues[self._treatment]

            self._results = {
                self._ATE: np.round(ate, 5),
                self._SE: np.round(se, 5),
                self._CI: (np.round(lower, 5), np.round(upper, 5)),
                self._PVALUE: np.round(p_value, 5),
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
            data = np.random.normal(
                loc=output[self._ATE], scale=output[self._SE], size=10_000
            )
            self._plot_result(data=data, effect=output[self._ATE], ci=output[self._CI])
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
