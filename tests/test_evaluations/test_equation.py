import unittest

import torch

from deep_macrofin.evaluations import Equation, HJBEquation


class TestEquationClasses(unittest.TestCase):

    def setUp(self):
        # Define available functions and variables for testing
        self.available_functions = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp
        }
        self.variables = {
            'x': torch.tensor([1.0, 2.0, 3.0]),
            'y': torch.tensor([4.0, 5.0, 6.0]),
            'z': torch.tensor([7.0, 8.0, 9.0])
        }

    def test_equation(self):
        equation = Equation("x = y", "Test Equation")
        result = equation.eval(self.available_functions, self.variables)
        expected_result = self.variables['y']
        self.assertTrue(torch.allclose(result, expected_result))

    def test_equation2(self):
        equation = Equation("x = y + z", "Test Equation")
        result = equation.eval(self.available_functions, self.variables)
        expected_result = self.variables['y'] + self.variables['z']
        self.assertTrue(torch.allclose(result, expected_result))

    def test_equation_with_derivative(self):
        z = torch.tensor([[1.0], [2.0], [3.0]])
        z.requires_grad_(True)
        q = 2 * z**2
        equation = Equation("q_z=deriv(q, z)", "eq1")
        result = equation.eval({}, {"q": q, "z": z})
        self.assertTrue(torch.allclose(result, torch.tensor([[4.0], [8.0], [12.0]])), f"Expected {torch.tensor([[4.0], [8.0], [12.0]])}, got {result}")
    
    def test_equation_with_derivative2(self):
        sv = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        sv.requires_grad_(True)
        variables = {}
        variables["z"] = sv[:, 0:1]
        variables["t"] = sv[:, 1:2]
        q = variables["z"] * variables["t"]
        variables["q"] = q
        equation = Equation("q_z=deriv(q, z)", "eq1")
        equation2 = Equation("q_t=deriv(q, t)", "eq2")
        result1 = equation.eval({}, variables)
        result2 = equation2.eval({}, variables)
        self.assertTrue(torch.allclose(result1, torch.tensor([[3.0], [2.0], [1.0]])), f"Expected {torch.tensor([[3.0], [2.0], [1.0]])}, got {result1}")
        self.assertTrue(torch.allclose(result2, torch.tensor([[1.0], [2.0], [3.0]])), f"Expected {torch.tensor([[1.0], [2.0], [3.0]])}, got {result2}")

    def test_hjb_equation(self):
        hjb_equation = HJBEquation("x - y", "Test HJB Equation")
        result = hjb_equation.eval(self.available_functions, self.variables)
        expected_result = torch.mean(torch.square(self.variables['x'] - self.variables['y']))
        self.assertTrue(torch.allclose(result, expected_result))

    def test_hjb_equation_complex_1(self):
        hjb_equation = HJBEquation("x + y - sin(z)", "Test HJB Equation Complex 1")
        result = hjb_equation.eval(self.available_functions, self.variables)
        expected_result = torch.mean(
            torch.square(self.variables['x'] + self.variables['y'] - torch.sin(self.variables['z'])))
        self.assertTrue(torch.allclose(result, expected_result))

    def test_hjb_equation_complex_2(self):
        hjb_equation = HJBEquation("exp(x) - y * z", "Test HJB Equation Complex 2")
        result = hjb_equation.eval(self.available_functions, self.variables)
        expected_result = torch.mean(
            torch.square(torch.exp(self.variables['x']) - self.variables['y'] * self.variables['z']))
        self.assertTrue(torch.allclose(result, expected_result))


if __name__ == '__main__':
    unittest.main()