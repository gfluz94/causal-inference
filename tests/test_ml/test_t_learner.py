import pandas as pd
import numpy as np
import pytest

from causal_inference.ml import TLearner
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class TestTLearner:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = TLearner(
                data=dummy_ate_data_for_propensity_score_matching,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                covariates=1.0,
            )

    def test__split_data_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = TLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
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
        model = TLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (
            model._model_T0 is not None and model._model_T1 is not None
        ) == model_not_None

    def test_predict_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = TLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            min_child_samples=1,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = model.predict(dummy_ate_data_for_propensity_score_matching)

    def test_fit_and_predict_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = TLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            min_child_samples=1,
        )
        model.fit()
        predictions = model.predict(dummy_ate_data_for_propensity_score_matching)

        # EXPECTED
        expected_predictions = np.array(
            [
                -0.3333894874418739,
                -0.3333894874418739,
                9.221648144163993,
                -0.3333894874418739,
                11.33309058875811,
                -10.499662687732135,
                -0.00012327868639694373,
                5.8415200975048265e-05,
                -0.9991821724284407,
                -0.00012327868639694373,
                8.721661406466504,
                8.33317027295479,
                5.8415200975048265e-05,
                10.333117150163071,
                5.8415200975048265e-05,
                11.33309058875811,
                8.721661406466504,
                -0.3333894874418739,
                11.33309058875811,
                8.721661406466504,
                9.221648144163993,
                -1.278072825363136,
                10.333117150163071,
                5.8415200975048265e-05,
                -10.499662687732135,
                9.221648144163993,
                -10.499662687732135,
                10.333117150163071,
                -10.499662687732135,
                8.721661406466504,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_train_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = TLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            min_child_samples=1,
        )
        model.fit()
        predictions = model.predict_train()

        # EXPECTED
        expected_predictions = np.array(
            [
                5.8415200975048265e-05,
                10.333117150163071,
                5.8415200975048265e-05,
                -0.00012327868639694373,
                8.33317027295479,
                10.333117150163071,
                -0.3333894874418739,
                8.721661406466504,
                -0.3333894874418739,
                -10.499662687732135,
                8.721661406466504,
                9.221648144163993,
                -10.499662687732135,
                11.33309058875811,
                11.33309058875811,
                -10.499662687732135,
                -0.00012327868639694373,
                -0.9991821724284407,
                9.221648144163993,
                -0.3333894874418739,
                -0.3333894874418739,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_test_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = TLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            min_child_samples=1,
        )
        model.fit()
        predictions = model.predict_test()

        # EXPECTED
        expected_predictions = np.array(
            [
                5.8415200975048265e-05,
                -10.499662687732135,
                8.721661406466504,
                5.8415200975048265e-05,
                -1.278072825363136,
                10.333117150163071,
                8.721661406466504,
                11.33309058875811,
                9.221648144163993,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)
