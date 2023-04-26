class InvalidDataFormatForCovariates(Exception):
    """Exception raised when `covariates` are neither List[str] or str."""


class ModelNotFittedYet(Exception):
    """Exception raised when results are requested, but model has not been fitted yet."""


class ATECannotBeEstimatedForHeterogeneousCase(Exception):
    """Exception raised when user attempts to estimate ATE for heterogeneous setting."""


class CATECannotBeEstimatedForHomogeneousCase(Exception):
    """Exception raised when user attempts to estimate CATE for homogeneous setting."""
