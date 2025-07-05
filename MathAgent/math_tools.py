from langchain_core.tools import tool

from typing import Annotated, List, Dict
import numexpr

@tool
def calculator(expression: str) -> str:
    """Вычислите выражение с помощью библиотеки Python numexpr.
    Используй меня если нужно вычислить какое-то выражение.

    Выражение должно быть математическим выражением в одну строку, которое решает задачу.
    Примеры:
    "37593 * 67" для "37593 умножить на 67"
    "37593**(1/5)" для "37593^(1/5)"
    "sqrt(4)" для "корень из 4"

    :param expression: str - арифметическое выражение, состоящее из чисел и операций +, -, /, *, **, (, ), log, sqrt

    return:
    решение арифметического выражения
    """
    return str(
        numexpr.evaluate(
            expression.strip()
        )
    )


@tool
def binary_to_decimal(binary_number: str):
    """
    Перевод числа из двоичной системы счисления в десятичную.

    :param binary_number": число в двоичной системе счисления

    return: число в десятичной системе счисления
    """

    return int(binary_number, 2)


@tool
def decimal_to_binary(decimal_number: int):
    """
    Перевод чисел из десятичной системы счисления в двоичную

    :param decimal_number: число в десятичной системе счисления

    return: число в двоичной системе счисления
    """
    return int(bin(decimal_number)[2:])


@tool
def general_response(query: str) -> str:
    """
    Генерация ответа на общие вопросы и запросы, которые не попадают под другие инструменты.
    Используется, когда запрос требует естественно-языкового ответа, объяснения, совета или не связан с вычислениями/конвертацией чисел.
    Отвечай

    :param query: Текст запроса пользователя
    :return: Ответ, сгенерированный языковой моделью
    """
    return query
