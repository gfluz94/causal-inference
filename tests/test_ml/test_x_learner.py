import pandas as pd
import numpy as np
import pytest

from causal_inference.ml import XLearner
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class TestXLearner:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = XLearner(
                data=dummy_ate_data_for_propensity_score_matching,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                covariates_numerical=1.0,
                covariates_categorical=self._CATEGORICAL_COVARIATE,
            )

    def test__split_data_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = XLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates_numerical=self._NUMERICAL_COVARIATE,
            covariates_categorical=self._CATEGORICAL_COVARIATE,
        )
        train, test = model._split_data()
        train_indices = train.index.values
        test_indices = test.index.values

        # EXPECTED
        expected_train_indices = np.array(
            [
                5586,
                5492,
                6022,
                1409,
                2751,
                7319,
                3949,
                4354,
                4509,
                517,
                6094,
                1198,
                3216,
                7902,
                7445,
                936,
                5161,
                2316,
                1371,
                6328,
                4495,
            ]
        )
        expected_test_indices = np.array(
            [3597, 142, 3331, 1252, 3155, 3591, 736, 1823, 7609]
        )

        # ASSERT
        np.testing.assert_array_equal(train_indices, expected_train_indices)
        np.testing.assert_array_equal(test_indices, expected_test_indices)

    def test_fit_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = XLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates_numerical=self._NUMERICAL_COVARIATE,
            covariates_categorical=self._CATEGORICAL_COVARIATE,
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (
            model._model_T0 is not None
            and model._model_T1 is not None
            and model._model_tau0 is not None
            and model._model_tau1 is not None
            and model._model_ps is not None
        ) == model_not_None

    def test_predict_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = XLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates_numerical=self._NUMERICAL_COVARIATE,
            covariates_categorical=self._CATEGORICAL_COVARIATE,
            min_child_samples=1,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = model.predict(dummy_ate_data_for_propensity_score_matching)

    def test_fit_and_predict_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = XLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates_numerical=self._NUMERICAL_COVARIATE,
            covariates_categorical=self._CATEGORICAL_COVARIATE,
            min_child_samples=1,
        )
        model.fit()
        predictions = model.predict(dummy_ate_data_for_propensity_score_matching)

        # EXPECTED
        expected_predictions = np.array(
            [
                -0.3332947897828198,
                -0.33328109423118585,
                9.716104605354726,
                -0.3332947897828198,
                10.60524724515055,
                -10.407894028988505,
                -0.2580175588002166,
                -6.536019077903056,
                -0.41496647028270783,
                -0.24892886870323802,
                9.43176144141883,
                3.4320849791703756,
                -8.053265586369522,
                10.160537391466306,
                -6.674754010906232,
                10.60524724515055,
                9.50594799886093,
                -0.33328109423118585,
                10.586482826871691,
                9.50594799886093,
                9.70232130264112,
                1.2575381872039957,
                10.18418970678255,
                -7.348270749068636,
                -10.413178371797686,
                9.74705162732975,
                -10.399961070467679,
                10.165279652434245,
                -10.399961070467679,
                9.46219234983886,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_train_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = XLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates_numerical=self._NUMERICAL_COVARIATE,
            covariates_categorical=self._CATEGORICAL_COVARIATE,
            min_child_samples=1,
        )
        model.fit()
        predictions = model.predict_train()

        # EXPECTED
        expected_predictions = np.array(
            [
                -6.536019077903056,
                10.18418970678255,
                -6.674754010906232,
                -0.2580175588002166,
                3.4320849791703756,
                10.160537391466306,
                -0.3332947897828198,
                9.50594799886093,
                -0.33328109423118585,
                -10.399961070467679,
                9.46219234983886,
                9.70232130264112,
                -10.407894028988505,
                10.60524724515055,
                10.586482826871691,
                -10.399961070467679,
                -0.24892886870323802,
                -0.41496647028270783,
                9.74705162732975,
                -0.3332947897828198,
                -0.33328109423118585,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_test_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = XLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates_numerical=self._NUMERICAL_COVARIATE,
            covariates_categorical=self._CATEGORICAL_COVARIATE,
            min_child_samples=1,
        )
        model.fit()
        predictions = model.predict_test()

        # EXPECTED
        expected_predictions = np.array(
            [
                -7.348270749068636,
                -10.413178371797686,
                9.50594799886093,
                -8.053265586369522,
                1.2575381872039957,
                10.165279652434245,
                9.43176144141883,
                10.60524724515055,
                9.716104605354726,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)
