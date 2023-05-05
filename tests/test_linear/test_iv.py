import pandas as pd
import pytest

from causal_inference.linear import IVEstimator
from causal_inference._exceptions.iv import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class TestIVEstimator:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _INSTRUMENT = "Z"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___InstrumentsNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_instrumental_variables: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = IVEstimator(
                data=dummy_ate_data_for_instrumental_variables,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                instruments=1.0,
                numerical_covariates=self._NUMERICAL_COVARIATE,
                categorical_covariates=self._CATEGORICAL_COVARIATE,
            )

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_instrumental_variables: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = IVEstimator(
                data=dummy_ate_data_for_instrumental_variables,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                instruments=self._INSTRUMENT,
                numerical_covariates=1.0,
                categorical_covariates=1.0,
            )

    def test_fit_RunsSuccessfully(
        self, dummy_ate_data_for_instrumental_variables: pd.DataFrame
    ):
        # OUTPUT
        model = IVEstimator(
            data=dummy_ate_data_for_instrumental_variables,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            instruments=self._INSTRUMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (model._model is not None) == model_not_None

    def test__get_results_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_instrumental_variables: pd.DataFrame
    ):
        model = IVEstimator(
            data=dummy_ate_data_for_instrumental_variables,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            instruments=self._INSTRUMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        with pytest.raises(ModelNotFittedYet):
            model._get_results()

    def test_fit_and__get_results_RunSuccessfully(
        self, dummy_ate_data_for_instrumental_variables: pd.DataFrame
    ):
        # OUTPUT
        model = IVEstimator(
            data=dummy_ate_data_for_instrumental_variables,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            instruments=self._INSTRUMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()
        model._get_results()
        output = model._results

        # EXPECTED
        results = results = {
            "ATE": -0.3567190568337537,
            "SE": 2.46882425861423,
            "CI": (-5.1955256878764455, 4.482087574208938),
            "p-VALUE": 0.8851139874955676,
        }

        # ASSERTION
        assert output == results

    def test_estimate_ate_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_instrumental_variables: pd.DataFrame
    ):
        model = IVEstimator(
            data=dummy_ate_data_for_instrumental_variables,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            instruments=self._INSTRUMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = model.estimate_ate()

    def test_fit_and_estimate_ate_RunSuccessfully(
        self, dummy_ate_data_for_instrumental_variables: pd.DataFrame
    ):
        # OUTPUT
        model = IVEstimator(
            data=dummy_ate_data_for_instrumental_variables,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            instruments=self._INSTRUMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()
        output = model.estimate_ate(plot_result=False)

        # EXPECTED
        results = {
            "ATE": -0.3567190568337537,
            "SE": 2.46882425861423,
            "CI": (-5.1955256878764455, 4.482087574208938),
            "p-VALUE": 0.8851139874955676,
        }

        # ASSERTION
        assert output == results
