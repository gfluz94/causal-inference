from typing import Any, List


def _check_input_validity(input_variable: Any, exception: Exception) -> List[str]:
    """Auxiliary function that checks whether the input belongs to one of the allowed types.

    Args:
        input_variable (Any): The input variable - a parameter passed to the class.

    Raises:
        InvalidDataFormatForInputs: Raised when `input_variable` is neither a list nor a string.

    Returns:
        List[str]: List of str.
    """
    if isinstance(input_variable, str):
        return [input_variable]
    elif isinstance(input_variable, list):
        return input_variable
    else:
        raise exception("Inputs must be either List[str] or str!")
