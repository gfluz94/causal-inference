import pandas as pd
import pytest

from causal_inference.linear import DiffInDiffEstimator
from causal_inference._exceptions.diff_in_diff import (
    ModelNotFittedYet,
)


class TestDiffInDiffEstimator:
    _OUTCOME = "Y"
    _TREATMENT = "D"
    _TIME_DIMENSION = "T"

    def test___init___RunsSuccessfully(
        self, dummy_ate_data_for_diff_in_diff: pd.DataFrame
    ):
        # OUTPUT
        model = DiffInDiffEstimator(
            data=dummy_ate_data_for_diff_in_diff,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            time_dimension=self._TIME_DIMENSION,
        )
        is_model_trained = model._fitted

        # EXPECTED
        expected_is_model_trained = False

        # ASSERT
        assert is_model_trained == expected_is_model_trained

    def test_fit_RunsSuccessfully(self, dummy_ate_data_for_diff_in_diff: pd.DataFrame):
        # OUTPUT
        model = DiffInDiffEstimator(
            data=dummy_ate_data_for_diff_in_diff,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            time_dimension=self._TIME_DIMENSION,
        )
        model.fit()
        is_model_trained = model._fitted

        # EXPECTED
        expected_is_model_trained = True

        # ASSERT
        assert is_model_trained == expected_is_model_trained

    def test__get_results_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_diff_in_diff: pd.DataFrame
    ):
        model = DiffInDiffEstimator(
            data=dummy_ate_data_for_diff_in_diff,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            time_dimension=self._TIME_DIMENSION,
        )
        with pytest.raises(ModelNotFittedYet):
            model._get_results()

    def test_fit_and__get_results_RunSuccessfully(
        self, dummy_ate_data_for_diff_in_diff: pd.DataFrame
    ):
        # OUTPUT
        model = DiffInDiffEstimator(
            data=dummy_ate_data_for_diff_in_diff,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            time_dimension=self._TIME_DIMENSION,
        )
        model.fit()
        model._get_results()
        output = model._results

        # EXPECTED
        results = {
            "ATE": 4.48295,
            "SE": 7.72461,
            "CI": (-11.3952, 20.36111),
            "p-VALUE": 0.56668,
        }

        # ASSERTION
        assert output == results

    def test_estimate_ate_RaisesModelNotFittedYet(
        self, dummy_ate_data_for_diff_in_diff: pd.DataFrame
    ):
        model = DiffInDiffEstimator(
            data=dummy_ate_data_for_diff_in_diff,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            time_dimension=self._TIME_DIMENSION,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = model.estimate_ate()

    def test_fit_and_estimate_ate_RunSuccessfully(
        self, dummy_ate_data_for_diff_in_diff: pd.DataFrame
    ):
        # OUTPUT
        model = DiffInDiffEstimator(
            data=dummy_ate_data_for_diff_in_diff,
            outcome=self._OUTCOME,
            treatment=self._TREATMENT,
            time_dimension=self._TIME_DIMENSION,
        )
        model.fit()
        model._get_results()
        output = model._results

        # EXPECTED
        results = {
            "ATE": 4.48295,
            "SE": 7.72461,
            "CI": (-11.3952, 20.36111),
            "p-VALUE": 0.56668,
        }

        # ASSERTION
        assert output == results
