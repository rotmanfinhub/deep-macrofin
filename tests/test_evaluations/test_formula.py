import unittest

import torch

from deepMacroFin.evaluations.formula import Formula


class TestFormulaEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LOCAL_DICT = {
            'simple_add': lambda X: X + 1,
            'simple_mul': lambda Y: Y * 2,
        }
        cls.variables = {
            "X": torch.tensor([1.0]),
            "Y": torch.tensor([2.0]),
            "a": torch.tensor([3.0]),
            "b": torch.tensor([4.0]),
            "q": torch.tensor([3.0]),
            "kappa": torch.tensor([1.0]),
            "iota": torch.tensor([0.5])
        }

    def test_simple_addition(self):
        formula_str = 'X + 1'
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([2.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_simple_multiplication(self):
        formula_str = 'Y * 2'
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([4.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_combined_operations(self):
        formula_str = 'X + 1 + Y * 2'
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([6.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_complex_function1(self):
        formula_str = '(q**2 - 1) / (2 * kappa)'
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([4.0])  # (3^2 - 1) / (2 * 1) = 4
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_complex_function2(self):
        formula_str = '1 / kappa * ((1 + 2 * kappa * iota)**0.5 - 1)'
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([0.4142])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4), f"Expected {expected_result}, got {result}") # allow some tolerance

    def test_elementary_function(self):
        formula_str = "sin(X) + cos(Y)"
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.sin(self.variables["X"]) + torch.cos(self.variables["Y"])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4), f"Expected {expected_result}, got {result}") # allow some tolerance

    def test_elementary_function2(self):
        formula_str = "exp(X) * Y"
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.exp(self.variables["X"]) * self.variables["Y"]
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4), f"Expected {expected_result}, got {result}") # allow some tolerance

    def test_elementary_function2(self):
        formula_str = "log(X) + exp(Y)"
        formula = Formula(formula_str, 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.log(self.variables["X"]) + torch.exp(self.variables["Y"])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4), f"Expected {expected_result}, got {result}") # allow some tolerance

if __name__ == '__main__':
    unittest.main()

