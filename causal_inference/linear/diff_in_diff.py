__all__ = ["DiffInDiffEstimator"]

from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


from causal_inference._exceptions.diff_in_diff import (
    ModelNotFittedYet,
)


class DiffInDiffEstimator(object):
    _ATE = "ATE"
    _SE = "SE"
    _CI = "CI"
    _PVALUE = "p-VALUE"

    def __init__(
        self, data: pd.DataFrame, outcome: str, treatment: str, time_dimension: str
    ) -> None:
        self._data = data
        self._outcome = outcome
        self._treatment = treatment
        self._time_dimension = time_dimension

        self._fitted = False
        self._model = None
        self._results = None

    def fit(self, plot_intervention: bool = False) -> None:
        self._model = smf.ols(
            formula=f"{self._outcome} ~ {self._treatment}*{self._time_dimension}",
            data=self._data,
        ).fit()
        if plot_intervention:
            self._plot_with_counterfactual()
        self._fitted = True

    def _get_results(self) -> Dict[str, Any]:
        if not self._fitted:
            raise ModelNotFittedYet("Model needs to be fitted first!")

        if self._results is None:
            parameter = f"{self._treatment}:{self._time_dimension}"
            ate = self._model.params[parameter]
            se = self._model.bse[parameter]
            lower_ci, upper_ci = self._model.conf_int().loc[parameter].values
            p_value = self._model.pvalues[parameter]
            self._results = {
                self._ATE: np.round(ate, 5),
                self._SE: np.round(se, 5),
                self._CI: (np.round(lower_ci, 5), np.round(upper_ci, 5)),
                self._PVALUE: np.round(p_value, 5),
            }

        return self._results

    def estimate_ate(self, plot_result: bool = False) -> Dict[str, Any]:
        output = self._get_results()
        if plot_result:
            np.random.seed(99)
            self._plot_result(
                data=np.random.normal(
                    loc=output[self._ATE], scale=output[self._SE], size=10_000
                ),
                effect=output[self._ATE],
                ci=output[self._CI],
            )
        return output

    def _plot_with_counterfactual(self) -> None:
        control = "Control"
        treated = "Treated"
        style.use("fivethirtyeight")
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
        agg_df = pd.pivot_table(
            data=self._data,
            index=self._time_dimension,
            columns=self._treatment,
            values=self._outcome,
            aggfunc=np.mean,
        ).rename(columns={0: control, 1: treated})

        treated_baseline = agg_df.loc[0, treated]
        treated_outcome = agg_df.loc[1, treated]
        treated_counterfactual = (
            treated_baseline + self._model.params[self._time_dimension]
        )

        ax.arrow(
            x=1,
            y=treated_counterfactual,
            dx=0,
            dy=(treated_outcome - treated_counterfactual),
            width=0.01,
            color="green",
            alpha=0.5,
            label=self._ATE,
        )
        agg_df.plot(ax=ax)
        ax.plot(
            [0, 1],
            [treated_baseline, treated_counterfactual],
            linestyle="--",
            color="black",
            alpha=0.3,
            label="Counterfactual",
        )

        plt.legend()
        plt.show()

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
