import unittest
import torch
from deepMacroFin.models.formula import Formula


class TestFormulaEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LOCAL_DICT = {
            'simple_add': lambda X: X + 1,
            'simple_mul': lambda Y: Y * 2,
            'complex_func1': lambda q, kappa: (q**2 - 1) / (2 * kappa),
            'complex_func2': lambda kappa, iota: 1 / kappa * ((1 + 2 * kappa * iota)**0.5 - 1)
        }
        cls.variables = {
            "X": torch.tensor([1.0]),
            "Y": torch.tensor([2.0]),
            "q": torch.tensor([3.0]),
            "kappa": torch.tensor([1.0]),
            "iota": torch.tensor([0.5])
        }

    def test_simple_addition(self):
        formula_str = 'simple_add(X)'
        formula = Formula(formula_str, ["X"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([2.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_simple_multiplication(self):
        formula_str = 'simple_mul(Y)'
        formula = Formula(formula_str, ["Y"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([4.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_combined_operations(self):
        formula_str = 'simple_add(X) + simple_mul(Y)'
        formula = Formula(formula_str, ["X", "Y"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([6.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_complex_function1(self):
        formula_str = 'complex_func1(q, kappa)'
        formula = Formula(formula_str, ["q", "kappa"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([4.0])  # (3^2 - 1) / (2 * 1) = 4
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_complex_function2(self):
        formula_str = 'complex_func2(kappa, iota)'
        formula = Formula(formula_str, ["kappa", "iota"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([0.4142])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4), f"Expected {expected_result}, got {result}") # allow small error tolerance in the result


if __name__ == '__main__':
    unittest.main()
