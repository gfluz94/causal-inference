class InvalidDataFormatForInputs(Exception):
    """Exception raised when inputs are neither List[str] or str."""


class ModelNotFittedYet(Exception):
    """Exception raised when results are requested, but model has not been fitted yet."""


class NoCovariatesAvailable(Exception):
    """Exception raised when no covariates are passed for Propensity Score Matching."""
