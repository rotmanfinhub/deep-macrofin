import unittest
import torch
from deepMacroFin.models.formula import Formula


class TestFormulaEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LOCAL_DICT = {
            'simple_add': lambda vars: vars["X"] + 1,
            'simple_mul': lambda vars: vars["Y"] * 2,
        }
        cls.variables = {
            "X": torch.tensor([1.0]),
            "Y": torch.tensor([2.0])
        }

    def test_simple_addition(self):
        formula_str = 'simple_add(vars)'
        formula = Formula(formula_str, ["X"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([2.0])
        self.assertTrue(torch.equal(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_simple_multiplication(self):
        formula_str = 'simple_mul(vars)'
        formula = Formula(formula_str, ["Y"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([4.0])
        self.assertTrue(torch.equal(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_combined_operations(self):
        formula_str = 'simple_add(vars) + simple_mul(vars)'
        formula = Formula(formula_str, ["X", "Y"], 'eval')
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([6.0])
        self.assertTrue(torch.equal(result, expected_result), f"Expected {expected_result}, got {result}")


if __name__ == '__main__':
    unittest.main()
