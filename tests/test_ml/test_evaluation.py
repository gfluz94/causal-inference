import pandas as pd
import numpy as np
import pytest

from causal_inference.ml import SLearner, CumulativeGainEvaluator
from causal_inference._exceptions.ml import EvaluatorNotFittedYet


class TestCumulativeGainEvaluator:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = SLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
        )
        train, test = model._split_data()
        evaluator = CumulativeGainEvaluator(
            train=train,
            test=test,
            model=model,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            model_name="SEP",
        )
        is_fitted = evaluator._fitted

        # EXPECTED
        expected_is_fitted = False

        # ASSERT
        assert is_fitted == expected_is_fitted

    def test_fit_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = SLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
        )
        train, test = model._split_data()
        model.fit()
        evaluator = CumulativeGainEvaluator(
            train=train,
            test=test,
            model=model,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            model_name="SEP",
        )
        evaluator.fit()
        is_fitted = evaluator._fitted

        # EXPECTED
        expected_is_fitted = True

        # ASSERT
        assert is_fitted == expected_is_fitted

    def test_predict_RaisesEvaluatorNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = SLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
        )
        train, test = model._split_data()
        model.fit()
        evaluator = CumulativeGainEvaluator(
            train=train,
            test=test,
            model=model,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            model_name="SEP",
        )
        with pytest.raises(EvaluatorNotFittedYet):
            _ = evaluator.eval()

    def test_fit_and_predict_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = SLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
        )
        model.fit()
        train, test = model._split_data()
        evaluator = CumulativeGainEvaluator(
            train=train,
            test=test,
            model=model,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            model_name="SEP",
        )
        evaluator.fit()
        predicted_df_with_cumulative_gain = evaluator._compute_cumulative_gain(test)
        print(predicted_df_with_cumulative_gain.to_dict())

        # EXPECTED
        expected_df_with_cumulative_gain = pd.DataFrame(
            {
                "Xcat": {
                    3597: 1,
                    142: 1,
                    3331: 1,
                    1252: 1,
                    3155: 1,
                    3591: 0,
                    736: 1,
                    1823: 0,
                    7609: 1,
                },
                "Xnum": {
                    3597: 37,
                    142: 50,
                    3331: 66,
                    1252: 31,
                    3155: 58,
                    3591: 67,
                    736: 61,
                    1823: 70,
                    7609: 73,
                },
                "Y": {
                    3597: 20,
                    142: 10,
                    3331: 12,
                    1252: 12,
                    3155: 13,
                    3591: 10,
                    736: 16,
                    1823: 11,
                    7609: 10,
                },
                "T": {
                    3597: 0,
                    142: 0,
                    3331: 0,
                    1252: 1,
                    3155: 0,
                    3591: 0,
                    736: 1,
                    1823: 0,
                    7609: 1,
                },
                "cate": {
                    3597: 0.0,
                    142: 0.0,
                    3331: 0.0,
                    1252: 0.0,
                    3155: 0.0,
                    3591: 0.0,
                    736: 0.0,
                    1823: 0.0,
                    7609: 0.0,
                },
                "sample_frac": {
                    3597: 0.1111111111111111,
                    142: 0.2222222222222222,
                    3331: 0.3333333333333333,
                    1252: 0.4444444444444444,
                    3155: 0.5555555555555556,
                    3591: 0.6666666666666666,
                    736: 0.7777777777777778,
                    1823: 0.8888888888888888,
                    7609: 1.0,
                },
                "cumulative_elasticity": {
                    3597: np.nan,
                    142: np.nan,
                    3331: np.nan,
                    1252: -2.0,
                    3155: -1.7499999999999998,
                    3591: -0.9999999999999998,
                    736: 1.0000000000000002,
                    1823: 1.3333333333333333,
                    7609: -2.220446049250313e-16,
                },
                "cumulative_gain": {
                    3597: np.nan,
                    142: np.nan,
                    3331: np.nan,
                    1252: -0.8888888888888888,
                    3155: -0.9722222222222221,
                    3591: -0.6666666666666665,
                    736: 0.777777777777778,
                    1823: 1.1851851851851851,
                    7609: -2.220446049250313e-16,
                },
            }
        )

        # ASSERT
        pd.testing.assert_frame_equal(
            predicted_df_with_cumulative_gain, expected_df_with_cumulative_gain
        )
