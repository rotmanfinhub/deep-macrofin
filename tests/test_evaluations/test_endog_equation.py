import unittest

import torch
import torch.nn as nn

from deepMacroFin.evaluations import EndogEquation

class TestConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variables = {
            "x": torch.rand_like(torch.tensor([0.0])),
            "y": torch.rand_like(torch.tensor([0.0])),
            "x_t": torch.rand_like(torch.tensor([0.0])),
            "x_tt": torch.rand_like(torch.tensor([0.0])),
            "y_t": torch.rand_like(torch.tensor([0.0])),

            "X": torch.rand((10, 3)),
            "Y": torch.rand((10, 1)),

            "wia": torch.rand_like(torch.tensor([0.0])),
            "wha": torch.rand_like(torch.tensor([0.0])),
            "eta": torch.rand_like(torch.tensor([0.0])),

            "ci": torch.rand_like(torch.tensor([0.0])),
            "ch": torch.rand_like(torch.tensor([0.0])),

            "alpha": torch.rand_like(torch.tensor([0.0])),
            "iota": torch.rand_like(torch.tensor([0.0])),
        }
        cls.LOCAL_DICT = {
            "f": lambda x: x[:, 0] + x[:, 1] * x[:, 2]
        }
        cls.latex_mapping = {
            "q_t^a": "qa",
            r"\xi_t^i": "xii",
            r"\eta_t": "eta",

            "c_t^i": "ci",
            "c_t^h": "ch",

            r"w_t^{ia}": "wia",
            r"w_t^{ha}": "wha",

            r"\rho^i": "rhoi",
            r"\zeta^i": "zetai",
            r"\mu^{\xi i}": "muxii",
            r"\sigma^{a}": "siga",
            r"\alpha^a": "alpha",
            r"\iota_t^a": "iota",
        }

    def test_endogeq(self):
        eq = EndogEquation("x**2+2*x+1=0", "label")
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["x"] ** 2 + 2 * self.variables["x"] + 1))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_endogeq2(self):
        eq = EndogEquation("x_t + x_tt / 2 = y", "label")
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["x_t"] + self.variables["x_tt"] / 2  - self.variables["y"]))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_endogeq3(self):
        eq = EndogEquation("y==3", "label")
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["y"] - 3))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_endogeq_multidim(self):
        eq = EndogEquation("Y == 1", "label")
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["Y"] - 1))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_endogeq_multidim2(self):
        eq = EndogEquation("f(X) = 3", "label")
        result = eq.eval(self.LOCAL_DICT, self.variables)
        X = self.variables["X"]
        f_X = X[:, 0] + X[:, 1] * X[:, 2]
        expected_result = torch.mean(torch.square(f_X - 3))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")


    def test_endogeq_latex(self):
        eq = EndogEquation(r"$x^2 = 1$", "label", self.latex_mapping)
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["x"] ** 2 - 1))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_endogeq_latex2(self):
        eq = EndogEquation(r"$\frac{\partial x}{\partial t} - 2*x + 1 = y$", "label", self.latex_mapping)
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["x_t"] - 2 * self.variables["x"] + 1 - self.variables["y"]))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_endogeq_latex3(self):
        eq = EndogEquation(r"$1=w_t^{ia} * \eta_t + w_t^{ha} * (1-\eta_t)$", "label", self.latex_mapping)
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["wia"] * self.variables["eta"] + self.variables["wha"] * (1 - self.variables["eta"]) - 1))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_endogeq_latex4(self):
        eq = EndogEquation(r"$\alpha^a -\iota_t^a &= c_t^i * \eta_t + c_t^h * (1-\eta_t)$", "label", self.latex_mapping)
        result = eq.eval({}, self.variables)
        expected_result = torch.mean(torch.square(self.variables["ci"] * self.variables["eta"] + self.variables["ch"] * (1 - self.variables["eta"]) - (self.variables["alpha"] - self.variables["iota"])))
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

if __name__ == "__main__":
    unittest.main()