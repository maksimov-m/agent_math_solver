def preprocess_calculator_expression(expression):
    expression = (expression.replace("π", "3.14")
                  .replace("math.pi", "3.14")
                  .replace("math.", "")
                  .replace("^", "**")
                  .replace("pi", "3.14"))

    return expression
