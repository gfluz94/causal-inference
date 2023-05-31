from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class MetaLearner(ABC):

    def __init__(
        self,
        data: pd.DataFrame,
        test_size: float = 0.30,
        seed: int = 99,
    ) -> None:
        """Constructor method for SLearner.

        Args:
            data (pd.DataFrame): pandas dataframe containing treatment, outcome and covariates.
            test_size (float, optional): Test size for in-sample hold-out validation. Defaults to 0.30.
            seed (int, optional): Random seed for reproducibility. Defaults to 99.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.
        """
        self._data = data
        self._test_size = test_size
        self._seed = seed

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Method that splits the dataset into training and test sets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test sets, respectively.
        """
        return train_test_split(
            self._data, test_size=self._test_size, random_state=self._seed
        )
    
    @abstractmethod
    def fit(self) -> None:
        """Method that fits the estimator on the training set of the input data."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Method that predicts CATE for an input dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe containing covariates and treatment.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            np.ndarray: Numpy array containing CATE predictions.
        """
    
    def predict_train(self) -> np.ndarray:
        """Method that predicts CATE for the training set of input data.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            np.ndarray: Numpy array containing CATE predictions.
        """
        train, _ = self._split_data()
        return self.predict(train)

    def predict_test(self) -> pd.DataFrame:
        """Method that predicts CATE for the test set of input data.

        Raises:
            ModelNotFittedYet: Exception raised when results are requested, but model has not been fitted yet.

        Returns:
            np.ndarray: Numpy array containing CATE predictions.
        """
        _, test = self._split_data()
        return self.predict(test)