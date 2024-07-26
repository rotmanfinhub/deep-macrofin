import unittest

import torch

from deep_macrofin.evaluations import Comparator, Constraint, System, EndogEquation

class TestSystems(unittest.TestCase):
    def setUp(self):
        self.variables = {
            'x': torch.tensor([[1.0], [2.0], [1.0]]),
            'x2': torch.tensor([[2.0], [1.0], [2.0]]),
            'y': torch.tensor([[1.0], [1.0], [3.0]]),
        }

    def test_constraint_mask(self):
        constraint = Constraint("x", Comparator.EQ, "y", "const1")
        sys = System([constraint], "system1")
        mask = sys.compute_constraint_mask({}, self.variables)
        expected_result = torch.tensor([[1], [0], [0]])
        self.assertTrue(torch.allclose(mask, expected_result), f"Expected {expected_result}, got {mask}")

    def test_constraint_mask2(self):
        constraint = Constraint("x", Comparator.GEQ, "x2", "const1")
        sys = System([constraint], "system1")
        mask = sys.compute_constraint_mask({}, self.variables)
        expected_result = torch.tensor([[0], [1], [0]])
        self.assertTrue(torch.allclose(mask, expected_result), f"Expected {expected_result}, got {mask}")

    def test_constraint_mask3(self):
        constraint = Constraint("x", Comparator.LEQ, "x2", "const1")
        sys = System([constraint], "system1")
        mask = sys.compute_constraint_mask({}, self.variables)
        expected_result = torch.tensor([[1], [0], [1]])
        self.assertTrue(torch.allclose(mask, expected_result), f"Expected {expected_result}, got {mask}")

    def test_constraint_mask5(self):
        sys = System([Constraint("x", Comparator.GEQ, "x2", "const1"),
                      Constraint("x", Comparator.LEQ, "x2", "const2")], 
                     "system1")
        mask = sys.compute_constraint_mask({}, self.variables)
        expected_result = torch.tensor([[0], [0], [0]])
        self.assertTrue(torch.allclose(mask, expected_result), f"Expected {expected_result}, got {mask}")

    def test_constraint_mask6(self):
        sys = System([Constraint("x", Comparator.LT, "y", "const1")], 
                     "system1")
        mask = sys.compute_constraint_mask({}, self.variables)
        expected_result = torch.tensor([[0], [0], [1]])
        self.assertTrue(torch.allclose(mask, expected_result), f"Expected {expected_result}, got {mask}")

    def test_constraint_mask7(self):
        sys = System([Constraint("x", Comparator.LT, "y", "const1"),
                      Constraint("x", Comparator.LEQ, "x2", "const2")], 
                     "system1")
        mask = sys.compute_constraint_mask({}, self.variables)
        expected_result = torch.tensor([[0], [0], [1]])
        self.assertTrue(torch.allclose(mask, expected_result), f"Expected {expected_result}, got {mask}")

    def test_system_loss(self):
        sys = System([Constraint("x", Comparator.LT, "y", "const1"),
                      Constraint("x", Comparator.LEQ, "x2", "const2")], 
                     "system1")
        sys.add_endog_equation("x=y")
        result = sys.eval({}, self.variables)
        expected_result = torch.tensor([4.0]) # only the last batch element will be computed, (3-1)^2
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_system_loss2(self):
        sys = System([Constraint("x", Comparator.LEQ, "x2", "const1")], 
                     "system1")
        sys.add_endog_equation("x=y-1")
        result = sys.eval({}, self.variables)
        expected_result = torch.tensor([1.0]) # first and last batch elements will be computed, (1^2+1^2)/ 2
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_system_loss3(self):
        sys = System([Constraint("x", Comparator.LEQ, "x2", "const1")], 
                     "system1")
        sys.add_endog_equation("x=y-1")
        sys.add_constraint("x", Comparator.GEQ, "y", "const")
        result = sys.eval({}, self.variables)
        expected_result = torch.tensor([3.0]) # first and last batch elements will be computed, (1^2+1^2)/ 2 + 2^2/2
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

if __name__ == "__main__":
    unittest.main()