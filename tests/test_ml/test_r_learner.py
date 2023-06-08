import pandas as pd
import numpy as np
import pytest

from causal_inference.ml import RLearner
from causal_inference._exceptions.ml import (
    InvalidDataFormatForInputs,
    ModelNotFittedYet,
)


class TestRLearner:
    _OUTCOME = "Y"
    _TREATMENT = "T"
    _NUMERICAL_COVARIATE = "Xnum"
    _CATEGORICAL_COVARIATE = "Xcat"

    def test___init___CovariatesNeitherListNorStrRaisesInvalidDataFormatForInputs(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        with pytest.raises(InvalidDataFormatForInputs):
            _ = RLearner(
                data=dummy_ate_data_for_propensity_score_matching,
                outcome=self._OUTCOME,
                treatment=self._TREATMENT,
                covariates=1.0,
            )

    def test__split_data_RunsSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = RLearner(
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
        model = RLearner(
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
            model._model_Tres is not None
            and model._model_Yres is not None
            and model._model is not None
        ) == model_not_None

    def test_predict_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        model = RLearner(
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
        model = RLearner(
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
                -0.8224300483924096,
                -0.8224300483924096,
                3.1642546295771368,
                -0.8224300483924096,
                10.376141208784018,
                0.6495121390512839,
                -1.4804388124989025,
                5.220968120485068,
                -2.3715030418619527,
                -1.4804388124989025,
                0.6495121390512839,
                -1.4804388124989025,
                5.440839816770408,
                10.376141208784018,
                5.220968120485068,
                10.376141208784018,
                3.3520119259552787,
                -0.8224300483924096,
                6.602266257688131,
                3.3520119259552787,
                3.1642546295771368,
                0.6495121390512839,
                -1.4804388124989025,
                5.440839816770408,
                0.6495121390512839,
                3.1642546295771368,
                0.6495121390512839,
                10.376141208784018,
                0.6495121390512839,
                0.6495121390512839,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_train_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = RLearner(
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
                5.220968120485068,
                -1.480438812498903,
                5.220968120485068,
                -1.480438812498903,
                -1.480438812498903,
                10.376141208784015,
                -0.8224300483924101,
                3.352011925955277,
                -0.8224300483924101,
                0.6495121390512835,
                0.6495121390512835,
                3.164254629577135,
                0.6495121390512835,
                10.376141208784015,
                6.602266257688129,
                0.6495121390512835,
                -1.480438812498903,
                -2.371503041861953,
                3.164254629577135,
                -0.8224300483924101,
                -0.8224300483924101,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_fit_and_predict_test_RunSuccessfully(
        self, dummy_ate_data_for_propensity_score_matching: pd.DataFrame
    ):
        # OUTPUT
        model = RLearner(
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
                5.440839816770408,
                0.6495121390512835,
                3.352011925955277,
                5.440839816770408,
                0.6495121390512835,
                10.376141208784015,
                0.6495121390512835,
                10.376141208784015,
                3.164254629577135,
            ]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(predictions, expected_predictions)
