import unittest

import torch
import torch.nn as nn

from deepMacroFin.evaluations import AgentConditions, Comparator, EndogVarConditions

class TestConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LOCAL_DICT = {
            "f": lambda x: x**2 - 2*x,
            "g": lambda x: 2 ** x,
            "h": lambda x: x[:, 0] * x[:, 1],
        }

    def test_eq_condition(self):
        condition = AgentConditions("f", "f(SV)", {"SV": torch.zeros((1,1))}, Comparator.EQ, "1", {}, "condition1")
        eval_mse = condition.eval(self.LOCAL_DICT)
        expected_mse = torch.tensor([1.0]) # f(0) = 0, mse = 1
        self.assertTrue(torch.allclose(eval_mse, expected_mse), f"Expected {expected_mse}, got {eval_mse}")

    def test_eq_condition2(self):
        condition = AgentConditions("f", "f(SV)", {"SV": torch.zeros((1,1))}, Comparator.EQ, "f(SV)", {"SV": torch.ones((1,1))}, "condition1")
        eval_mse = condition.eval(self.LOCAL_DICT)
        expected_mse = torch.tensor([1.0]) # f(0) = 0, f(1)=-1, mse = 1
        self.assertTrue(torch.allclose(eval_mse, expected_mse), f"Expected {expected_mse}, got {eval_mse}")

    def test_eq_condition3(self):
        # batched multi-variable evaluation, although very rare
        lhs_x = torch.rand((10, 2))
        rhs_x = torch.ones((10, 2))
        condition = AgentConditions("h", "h(SV)", {"SV": lhs_x}, Comparator.EQ, "h(SV)", {"SV": rhs_x}, "condition1")
        eval_mse = condition.eval(self.LOCAL_DICT)
        expected_mse = torch.mean(torch.square(self.LOCAL_DICT["h"](lhs_x) - self.LOCAL_DICT["h"](rhs_x)))
        self.assertTrue(torch.allclose(eval_mse, expected_mse), f"Expected {expected_mse}, got {eval_mse}")
    

    def test_geq_condition(self):
        condition = EndogVarConditions("g", "g(SV)", {"SV": torch.zeros((1,1))}, Comparator.GEQ, "2", {}, "condition1")
        eval_mse = condition.eval(self.LOCAL_DICT)
        expected_mse = torch.tensor([1.0]) # f(0) = 1, mse = 1
        self.assertTrue(torch.allclose(eval_mse, expected_mse), f"Expected {expected_mse}, got {eval_mse}")

    def test_geq_condition2(self):
        condition = EndogVarConditions("g", "g(SV)", {"SV": torch.ones((1,1)) * 2}, Comparator.GEQ, "1", {}, "condition1")
        eval_mse = condition.eval(self.LOCAL_DICT)
        expected_mse = torch.tensor([0.0]) # f(2) = 4, mse = 0
        self.assertTrue(torch.allclose(eval_mse, expected_mse), f"Expected {expected_mse}, got {eval_mse}")

    def test_leq_condition(self):
        condition = EndogVarConditions("g", "g(SV)", {"SV": torch.zeros((1,1))}, Comparator.LEQ, "1", {}, "condition1")
        eval_mse = condition.eval(self.LOCAL_DICT)
        expected_mse = torch.tensor([0.0]) # f(0) = 0, mse = 0
        self.assertTrue(torch.allclose(eval_mse, expected_mse), f"Expected {expected_mse}, got {eval_mse}")

    def test_leq_condition2(self):
        condition = EndogVarConditions("g", "g(SV)", {"SV": torch.ones((1,1)) * 2}, Comparator.LEQ, "1", {}, "condition1")
        eval_mse = condition.eval(self.LOCAL_DICT)
        expected_mse = torch.tensor([9.0]) # f(2) = 4, mse = 3^2
        self.assertTrue(torch.allclose(eval_mse, expected_mse), f"Expected {expected_mse}, got {eval_mse}")

if __name__ == "__main__":
    unittest.main()