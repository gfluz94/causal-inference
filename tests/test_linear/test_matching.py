import pandas as pd
import numpy as np
import pytest

from causal_inference.linear import MatchingEstimator
from causal_inference.linear.matching import PropensityScoreModel
from causal_inference._exceptions.matching import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
    NoCovariatesAvailable,
)


class TestPropensityScoreModel:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = PropensityScoreModel(
                data=dummy_ate_data_for_propensity_score_matching,
                treatment=self._TREATMENT,
                numerical_covariates=1.0,
                categorical_covariates=1.0,
            )

    def test___init___CovariatesNotPassedRaisesNoCovariatesAvailable(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(NoCovariatesAvailable):
            _ = PropensityScoreModel(
                data=dummy_ate_data_for_propensity_score_matching,
                treatment=self._TREATMENT,
            )

    def test_fit_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = PropensityScoreModel(
            data=dummy_ate_data_for_propensity_score_matching,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (model._pipeline is not None) == model_not_None

    def test_get_propensity_scores_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = PropensityScoreModel(
            data=dummy_ate_data_for_propensity_score_matching,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()
        scores = model.get_propensity_scores()

        # EXPECTED
        expected_scores = np.array(
            [
                0.7732835033067533,
                0.8192784760562788,
                0.24100677350313293,
                0.7732835033067533,
                0.4163143657608916,
                0.46223626003544654,
                0.6149628126614154,
                0.55675894178265,
                0.7562271551396867,
                0.5807763551195685,
                0.35936410785385475,
                0.5691884930923962,
                0.6994082247230559,
                0.43952711944086686,
                0.5684276360655633,
                0.4163143657608916,
                0.3067779294239376,
                0.8192784760562788,
                0.4048390964743494,
                0.3067779294239376,
                0.2587784319829294,
                0.39272681632037426,
                0.49850492158737353,
                0.6364433179166642,
                0.4858780741620683,
                0.20032426762349909,
                0.42712181816748884,
                0.4512401543179326,
                0.42712181816748884,
                0.337833110815498,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(scores, expected_scores)


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
