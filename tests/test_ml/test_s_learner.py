import pandas as pd
import numpy as np
import pytest

from causal_inference.ml import SLearner
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class TestSLearner:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = SLearner(
                data=dummy_ate_data_for_propensity_score_matching,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                covariates=1.0,
            )

    def test__split_data_RunsSuccessfully(
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
        model = SLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
        )
        model.fit()

        # EXPECTED
        model_not_None = True

        # ASSERT
        assert (model._model is not None) == model_not_None

    def test_predict_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = SLearner(
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
        model = SLearner(
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
                -0.41286836867116605,
                -0.41286836867116605,
                0.6652884848173279,
                -0.41286836867116605,
                3.168262794171536,
                -9.692239880431458,
                -2.1435599945009525,
                -4.078974496660138,
                -0.6018935366430203,
                3.8251369064342686,
                0.9956950646018079,
                3.8251369064342686,
                -3.889949328688303,
                3.168262794171536,
                -4.078974496660138,
                3.168262794171536,
                0.3388209523390735,
                -0.41286836867116605,
                3.9100178484624486,
                0.3388209523390735,
                0.6652884848173279,
                0.9956950646018079,
                3.8251369064342686,
                -3.889949328688303,
                -9.692239880431458,
                0.6652884848173279,
                -9.692239880431458,
                3.168262794171536,
                -9.692239880431458,
                0.9956950646018079,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_train_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = SLearner(
            data=dummy_ate_data_for_propensity_score_matching,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            covariates=[self._CATEGORICAL_COVARIATE, self._NUMERICAL_COVARIATE],
            min_child_samples=1,
        )
        model.fit()
        predictions = model.predict_train()
        print(predictions.tolist())

        # EXPECTED
        expected_predictions = np.array(
            [
                -4.078974496660138,
                3.8251369064342686,
                -4.078974496660138,
                -2.1435599945009525,
                3.8251369064342686,
                3.168262794171536,
                -0.41286836867116605,
                0.3388209523390735,
                -0.41286836867116605,
                -9.692239880431458,
                0.9956950646018079,
                0.6652884848173279,
                -9.692239880431458,
                3.168262794171536,
                3.9100178484624486,
                -9.692239880431458,
                3.8251369064342686,
                -0.6018935366430203,
                0.6652884848173279,
                -0.41286836867116605,
                -0.41286836867116605,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_test_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = SLearner(
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
                -3.889949328688303,
                -9.692239880431458,
                0.3388209523390735,
                -3.889949328688303,
                0.9956950646018079,
                3.168262794171536,
                0.9956950646018079,
                3.168262794171536,
                0.6652884848173279,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)
