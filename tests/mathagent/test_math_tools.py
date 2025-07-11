import pytest
from MathAgent.math_tools import (calculator,
                                  general_response,
                                  decimal_to_binary,
                                  binary_to_decimal)


@pytest.mark.parametrize("expression, expected_answer", [("5 + 5", "10"),
                                                         ("37593 * 67", "2518731"),
                                                         ("sqrt(4)", "2.0"),
                                                         ("7 - 0", "7")
                                                         ])
def test_calculator(expression, expected_answer):
    assert calculator(expression) == expected_answer


def test_calculator_divided_zero():
    with pytest.raises(ZeroDivisionError):
        calculator("1 / 0")


