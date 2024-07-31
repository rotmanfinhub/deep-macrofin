import unittest

import torch

from deep_macrofin.evaluations import Comparator, Constraint, LossReductionMethod


class TestConstraints(unittest.TestCase):
    def setUp(self):
        # Define available functions and variables for testing
        self.available_functions = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp
        }
        self.variables = {
            'x': torch.tensor([1.0, 2.0, 3.0]),
            'x2': torch.tensor([1.0, 1.0, 1.0]),
            'y': torch.tensor([4.0, 5.0, 6.0]),
            'z': torch.tensor([7.0, 8.0, 9.0])
        }
    
    def test_constraint_equal(self):
        constraint = Constraint("x", Comparator.EQ, "y", "Test Constraint Equal")
        result = constraint.eval(self.available_functions, self.variables)
        expected_result = torch.mean(torch.square(self.variables['x'] - self.variables['y']))
        self.assertTrue(torch.allclose(result, expected_result))

    def test_constraint_greater(self):
        constraint = Constraint("x", Comparator.GEQ, "y", "Test Constraint Greater")
        result = constraint.eval(self.available_functions, self.variables)
        expected_result = torch.mean(torch.square(torch.relu(self.variables['y'] - self.variables['x'])))
        self.assertTrue(torch.allclose(result, expected_result))

    def test_constraint_less(self):
        constraint = Constraint("x", Comparator.LEQ, "y", "Test Constraint Less")
        result = constraint.eval(self.available_functions, self.variables)
        expected_result = torch.mean(torch.square(torch.relu(self.variables['x'] - self.variables['y'])))
        self.assertTrue(torch.allclose(result, expected_result))

    def test_constraint_eval_loss_reduction(self):
        constraint = Constraint("x", Comparator.GEQ, "y", "const")
        result = constraint.eval(self.available_functions, self.variables, loss_reduction=LossReductionMethod.MAE)
        expected_result = torch.mean(torch.abs(torch.relu(self.variables['y'] - self.variables['x'])))
        self.assertTrue(torch.allclose(result, expected_result))

    def test_constraint_eval_loss_reduction2(self):
        constraint = Constraint("x", Comparator.GEQ, "y", "const")
        result = constraint.eval(self.available_functions, self.variables, loss_reduction=LossReductionMethod.NONE)
        expected_result = torch.relu(self.variables['y'] - self.variables['x'])
        self.assertTrue(torch.allclose(result, expected_result))

    def test_constraint_eval_no_loss(self):
        constraint = Constraint("x", Comparator.GEQ, "y", "const")
        result = constraint.eval_no_loss(self.available_functions, self.variables)
        expected_result = torch.relu(self.variables['y'] - self.variables['x'])
        self.assertTrue(torch.allclose(result, expected_result))

    def test_constraint_eval_masked(self):
        constraint = Constraint("x", Comparator.LEQ, "x2", "const")
        result = constraint.eval_with_mask(self.available_functions, self.variables, mask=torch.Tensor([[0], [0], [1]]))
        expected_result = torch.tensor([4.0])
        self.assertTrue(torch.allclose(result, expected_result))

    def test_constraint_eval_masked2(self):
        constraint = Constraint("x", Comparator.LEQ, "x2", "const")
        result = constraint.eval_with_mask(self.available_functions, self.variables, mask=torch.Tensor([[0], [1], [0]]))
        expected_result = torch.tensor([1.0])
        self.assertTrue(torch.allclose(result, expected_result))

if __name__ == '__main__':
    unittest.main()