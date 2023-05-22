class ModelNotFittedYet(Exception):
    """Exception raised when results are requested, but model has not been fitted yet."""


class EvaluatorNotFittedYet(Exception):
    """Exception raised when results are requested, but evaluator object has not been fitted yet."""


class InvalidDataFormatForInputs(Exception):
    """Exception raised when inputs are neither List[str] or str."""
