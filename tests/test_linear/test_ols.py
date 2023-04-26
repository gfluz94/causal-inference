import pandas as pd
import pytest

from causal_inference.linear import OLSEstimator
from causal_inference._exceptions.ols import (
    InvalidDataFormatForCovariates,
    ModelNotFittedYet,
    ATECannotBeEstimatedForHeterogeneousCase,
    CATECannotBeEstimatedForHomogeneousCase,
)


class TestOLSEstimator:

    _OUTCOME = "Y"
    _TREATMENT = "T"
    _COVARIATES = ["X1", "X2"]

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForCovariates(
        self, dummy_ate_data: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForCovariates):
            _ = OLSEstimator(
                data=dummy_ate_data,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                covariates=1.0,
                heterogeneous=False,
            )

    def test_fit_RunsSuccessfullyWithHeterogeneousSetToFalse(
        self, dummy_ate_data: pd.DataFrame
    ):
        # OUTPUT
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=False,
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (model._model is not None) == model_not_None
        assert hasattr(model, "estimate_ate")

    def test_fit_RunsSuccessfullyWithHeterogeneousSetToTrue(
        self, dummy_ate_data: pd.DataFrame
    ):
        # OUTPUT
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=True,
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (model._model is not None) == model_not_None
        assert hasattr(model, "estimate_cate")

    def test__get_results_RaisesModelNotFittedYet(self, dummy_ate_data: pd.DataFrame):
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=False,
        )
        with pytest.raises(ModelNotFittedYet):
            model._get_results()

    def test_fit_and__get_results_RunSuccessfullyWithHeterogeneousSetToFalse(
        self, dummy_ate_data: pd.DataFrame
    ):
        # OUTPUT
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=False,
        )
        model.fit()
        model._get_results()
        output = model._results

        # EXPECTED
        results = {
            "ATE": 1766.8330085269008,
            "SE": 2496.6129359235974,
            "CI": (-3365.0283781606613, 6898.694395214463),
            "p-VALUE": 0.4854325249810071,
        }

        # ASSERTION
        assert output == results

    def test_fit_and_estimate_ate_RunSuccessfullyWithHeterogeneousSetToFalse(
        self, dummy_ate_data: pd.DataFrame
    ):
        # OUTPUT
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=False,
        )
        model.fit()
        output = model.estimate_ate()

        # EXPECTED
        results = {
            "ATE": 1766.8330085269008,
            "SE": 2496.6129359235974,
            "CI": (-3365.0283781606613, 6898.694395214463),
            "p-VALUE": 0.4854325249810071,
        }

        # ASSERTION
        assert output == results

    def test__estimate_ate_RaisesModelNotFittedYet(self, dummy_ate_data: pd.DataFrame):
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=False,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = model._estimate_ate()

    def test__estimate_ate_RaisesATECannotBeEstimatedForHeterogeneousCase(
        self, dummy_ate_data: pd.DataFrame
    ):
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=True,
        )
        model.fit()
        with pytest.raises(ATECannotBeEstimatedForHeterogeneousCase):
            _ = model._estimate_ate()

    def test__get_cis_for_heterogeneity_RaisesModelNotFittedYet(
        self, dummy_ate_data: pd.DataFrame
    ):
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=True,
        )
        with pytest.raises(ModelNotFittedYet):
            model._get_cis_for_heterogeneity()

    def test_fit_and__get_cis_for_heterogeneity_RunSuccessfullyWithHeterogeneousSetToTrue(
        self, dummy_ate_data: pd.DataFrame
    ):
        # OUTPUT
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=True,
        )
        model.fit()
        model._get_cis_for_heterogeneity()
        output = model._coeffs_interaction

        # EXPECTED
        results = {
            "X1": (-0.7251981133807204, 1.932804974935116),
            "X2": (-2509.5136029747755, 3180.8431008660427),
            "T": (-2509.513602972725, 3180.8431008663642),
        }

        # ASSERTION
        assert output == results

    def test_fit_and_estimate_cate_RunSuccessfullyWithHeterogeneousSetToTrue(
        self, dummy_ate_data: pd.DataFrame
    ):
        # OUTPUT
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=True,
        )
        model.fit()
        output = model.estimate_cate(covariates={"X1": 10_000, "X2": 1})

        # EXPECTED
        results = {
            "CATE": 6723.493121389662,
            "CI": (-7171.841112871476, 20744.21855545908),
        }

        # ASSERTION
        assert output == results

    def test__estimate_cate_RaisesModelNotFittedYet(self, dummy_ate_data: pd.DataFrame):
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=True,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = model._estimate_cate(covariates={"X1": 10_000, "X2": 1})

    def test__estimate_cate_RaisesCATECannotBeEstimatedForHomogeneousCase(
        self, dummy_ate_data: pd.DataFrame
    ):
        model = OLSEstimator(
            data=dummy_ate_data,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=self._COVARIATES,
            heterogeneous=False,
        )
        model.fit()
        with pytest.raises(CATECannotBeEstimatedForHomogeneousCase):
            _ = model._estimate_cate(covariates={})
