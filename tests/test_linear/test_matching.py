import pandas as pd
import pytest

from causal_inference.linear import MatchingEstimator
from causal_inference._exceptions.matching import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
    NoCovariatesAvailable,
)


class TestMatchingEstimator:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = MatchingEstimator(
                data=dummy_ate_data_for_propensity_score_matching,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                numerical_covariates=1.0,
                categorical_covariates=1.0,
            )

    def test_fit_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = MatchingEstimator(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (model._propensity_score_model is not None) == model_not_None

    def test__get_results_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = MatchingEstimator(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        with pytest.raises(ModelNotFittedYet):
            model._get_results()

    def test_fit_and__get_results_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = MatchingEstimator(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()
        model._get_results()
        output = model._results

        # EXPECTED
        results = {"ATE": 1.61343, "CI": (-11.17272, 13.46655)}

        # ASSERTION
        assert output == results

    def test_estimate_ate_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = MatchingEstimator(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = model.estimate_ate()

    def test_fit_and_estimate_ate_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = MatchingEstimator(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()
        output = model.estimate_ate(plot_result=False)

        # EXPECTED
        results = {"ATE": 1.61343, "CI": (-11.17272, 13.46655)}

        # ASSERTION
        assert output == results
