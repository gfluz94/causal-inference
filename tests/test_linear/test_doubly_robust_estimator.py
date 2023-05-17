import pandas as pd
import numpy as np
import pytest

from causal_inference.linear import DoublyRobustEstimator
from causal_inference.linear.doubly_robust_estimator import LinearRegressionModel
from causal_inference._exceptions.doubly_robust_estimator import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
    NoCovariatesAvailable,
)


class TestLinearRegressionModel:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = LinearRegressionModel(
                data=dummy_ate_data_for_propensity_score_matching,
                outcome=self._OUTCOME,
                numerical_covariates=1.0,
                categorical_covariates=1.0,
            )

    def test___init___CovariatesNotPassedRaisesNoCovariatesAvailable(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(NoCovariatesAvailable):
            _ = LinearRegressionModel(
                data=dummy_ate_data_for_propensity_score_matching,
                outcome=self._OUTCOME,
            )

    def test_fit_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = LinearRegressionModel(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (model._pipeline is not None) == model_not_None

    def test_predict_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = LinearRegressionModel(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()
        predictions = model.predict(
            dummy_ate_data_for_propensity_score_matching[
                [self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE]
            ]
        )
        print(predictions.tolist())

        # EXPECTED
        expected_predictions = np.array(
            [
                14.606899815223759,
                14.852501472459947,
                14.572037092004269,
                14.606899815223759,
                13.25609070042472,
                15.431642892330931,
                13.951962062593921,
                15.759111768645848,
                14.52503259614503,
                13.829161233975828,
                15.063240406476648,
                13.788227624436463,
                16.291248692657593,
                13.33795791950345,
                15.800045378185214,
                13.25609070042472,
                14.858572358779824,
                14.852501472459947,
                13.215157090885356,
                14.858572358779824,
                14.653904311083,
                15.186041235094741,
                13.542625967200273,
                16.045647035421403,
                15.51351011140966,
                14.367369044307447,
                15.308842063712836,
                13.378891529042814,
                15.308842063712836,
                14.981373187397917,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)


class TestDoublyRobustEstimator:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = DoublyRobustEstimator(
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
        model = DoublyRobustEstimator(
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
        assert (model._mu0_model is not None) == model_not_None
        assert (model._mu1_model is not None) == model_not_None

    def test__get_results_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = DoublyRobustEstimator(
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
        model = DoublyRobustEstimator(
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
        results = {"ATE": 2.00417, "CI": (-2.91328, 6.78148)}

        # ASSERTION
        assert output == results

    def test_estimate_ate_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = DoublyRobustEstimator(
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
        model = DoublyRobustEstimator(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            numerical_covariates=self._NUMERICAL_COVARIATE,
            categorical_covariates=self._CATEGORICAL_COVARIATE,
        )
        model.fit()
        output = model.estimate_ate(plot_result=False)

        # EXPECTED
        results = {"ATE": 2.00417, "CI": (-2.91328, 6.78148)}

        # ASSERTION
        assert output == results
