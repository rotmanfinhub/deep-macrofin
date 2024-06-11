import unittest

import torch

from deep_macrofin.evaluations.formula import Formula


class TestFormulaLatexEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LOCAL_DICT = {}
        cls.variables = {
            "X": torch.tensor([1.0]),
            "Y": torch.tensor([2.0]),
            "qa": torch.tensor([3.0]),
            "xii": torch.tensor([4.0]),
            "xih": torch.tensor([3.0]),
            "eta": torch.tensor([1.0]),
            "sigea": torch.tensor([0.5]),
            "qa_eta": torch.tensor([0.8]),
            "qa_etaeta": torch.tensor([2.0]),
            "xii_eta": torch.tensor([0.9]),
            "xii_etaeta": torch.tensor([3.0]),
            "xih_eta": torch.tensor([0.7]),
            "xih_etaeta": torch.tensor([2.5]),
            "mue": torch.tensor([0.6]),

            "muni": torch.rand_like(torch.tensor([0.0])),
            "munh": torch.rand_like(torch.tensor([0.0])),
            "signa": torch.rand_like(torch.tensor([0.0])),
            "signia": torch.rand_like(torch.tensor([0.0])),
            "sigxia": torch.rand_like(torch.tensor([0.0])),
            "signha": torch.rand_like(torch.tensor([0.0])),

            "gammai": torch.rand_like(torch.tensor([0.0])),
            "gammah": torch.rand_like(torch.tensor([0.0])),
            "siga": torch.rand_like(torch.tensor([0.0])),
            "ci": torch.rand_like(torch.tensor([0.0])),
            "ch": torch.rand_like(torch.tensor([0.0])),
            "wia": torch.rand_like(torch.tensor([0.0])),
            "wha": torch.rand_like(torch.tensor([0.0])),
            "rhoi": torch.rand_like(torch.tensor([0.0])),
            "zetai": torch.rand_like(torch.tensor([0.0])),
            "muxii": torch.rand_like(torch.tensor([0.0])),
            "sigqa": torch.rand_like(torch.tensor([0.0])),
        }
        cls.latex_var_mapping = {
            "q_t^a": "qa",
            r"\xi_t^i": "xii",
            r"\xi_t^h": "xih",
            r"\eta_t": "eta",
            r"\sigma^{\eta a}_t": "sigea",
            r"\sigma_t^{\eta a}": "sigea",
            r"\sigma_t^{qa}": "sigqa",
            r"\sigma^{qa}_t": "sigqa",
            r"\sigma_t^{\xi ia}": "sigxia",
            r"\sigma_t^{\xi ha}": "sigxha",
            r"\mu^{\eta}_t": "mue",

            r"\mu^{n i}_t": "muni",
            r"\mu^{n h}_t": "munh",
            r"\sigma^{na}_t": "signa",
            r"\sigma^{nia}_t": "signia",
            r"\sigma^{nha}_t": "signha",
            

            r"\gamma^i": "gammai",
            r"\gamma^h": "gammah",
            r"\sigma^a": "siga",

            "c_t^i": "ci",
            "c_t^h": "ch",

            r"w_t^{ia}": "wia",
            r"w_t^{ha}": "wha",

            r"\rho^i": "rhoi",
            r"\zeta^i": "zetai",
            r"\mu^{\xi i}": "muxii",
            r"\sigma^{a}": "siga"
        }

    def test_simple_addition(self):
        formula_str = r'$X + 1$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([2.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_simple_parsing(self):
        formula_str = r'$X^2 + 2*X + qa^Y$' # 1^2 + 2*1+3^2
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([12.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_parsing_fraction(self):
        formula_str = r'$\frac{qa}{Y}$' # 3/2
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([1.5])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_parsing_fraction2(self):
        formula_str = r'$\frac{qa}{1-\frac{Y}{qa}}$' # 3/(1-2/3)
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([9.0])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_parsing_fraction3(self):
        formula_str = r'$\frac{qa}{1-\frac{Y^4}{qa}}$' # 3/(1-2^4/3)=-9/13
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([-9/13])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")
    
    def test_latex_parsing_fraction4(self):
        formula_str = r'$\left(\frac{qa}{1-\frac{Y}{qa}}\right)^{1-q_t^a}$' # (3/(1-2/3))^{1-3}=1/81
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = torch.tensor([1/81])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_parsing_derivatives(self):
        formula_str = r'$\frac{\partial q_t^a}{\partial \eta_t}$' # qa_eta
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["qa_eta"]
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_parsing_fraction_with_derivatives(self):
        formula_str = r'$\frac{\frac{\partial q_t^a}{\partial \eta_t}}{1-\frac{\partial \xi_t^i}{\partial \eta_t}}$' # qa_eta/(1-xii_eta)
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["qa_eta"] / (1 - self.variables["xii_eta"])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula(self):
        formula_str = r'$\frac{\partial q_t^a}{\partial \eta_t} * \sigma^{\eta a}_t * \eta_t$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["qa_eta"] * self.variables["sigea"] * self.variables["eta"]
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula2(self):
        formula_str = r'$\frac{\partial \xi_t^i}{\partial \eta_t} * \sigma^{\eta a}_t * \eta_t$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["xii_eta"] * self.variables["sigea"] * self.variables["eta"]
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula3(self):
        formula_str = r'$\frac{\partial \xi_t^h}{\partial \eta_t} * \sigma^{\eta a}_t * \eta_t$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["xih_eta"] * self.variables["sigea"] * self.variables["eta"]
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula4(self):
        formula_str = r'$\frac{\partial q_t^a}{\partial \eta_t} * \mu^{\eta}_t * \eta_t + \frac{1}{2} * \frac{\partial^2 q_t^a}{\partial \eta_t^2} * (\sigma_t^{\eta a})^2 * \eta_t^2$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["qa_eta"] * self.variables["mue"] * self.variables["eta"] \
                        + 0.5 * self.variables["qa_etaeta"] * self.variables["sigea"] ** 2 * self.variables["eta"] ** 2
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula5(self):
        formula_str = r'$\frac{\partial \xi_t^i}{\partial \eta_t} * \mu^{\eta}_t * \eta_t + \frac{1}{2} * \frac{\partial^2 \xi_t^i}{\partial \eta_t^2} * (\sigma_t^{\eta a})^2 * \eta_t^2$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["xii_eta"] * self.variables["mue"] * self.variables["eta"] \
                        + 0.5 * self.variables["xii_etaeta"] * self.variables["sigea"] ** 2 * self.variables["eta"] ** 2
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula6(self):
        formula_str = r'$\frac{\partial \xi_t^h}{\partial \eta_t} * \mu^{\eta}_t * \eta_t + \frac{1}{2} * \frac{\partial^2 \xi_t^h}{\partial \eta_t^2} * (\sigma_t^{\eta a})^2 * \eta_t^2$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["xih_eta"] * self.variables["mue"] * self.variables["eta"] \
                        + 0.5 * self.variables["xih_etaeta"] * self.variables["sigea"] ** 2 * self.variables["eta"] ** 2
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula7(self):
        formula_str = r'$(1-\eta_t) * (\mu^{n i}_t - \mu^{n h}_t) +(\sigma^{na}_t)^2  - \sigma^{nia}_t * \sigma^{na}_t$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = (1 - self.variables["eta"]) * (self.variables["muni"] - self.variables["munh"]) + self.variables["signa"] ** 2 \
            - self.variables["signia"] * self.variables["signa"]
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula8(self):
        formula_str = r'$(1-\eta_t) * \left( \sigma^{nia}_t - \sigma^{nha}_t\right)$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = (1 - self.variables["eta"]) * (self.variables["signia"] - self.variables["signha"])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula9(self):
        formula_str = r'$\gamma^i * w_t^{ia} * ( \sigma^a  + \sigma^{qa}_t)^2 + (1-\gamma^i) * \sigma_t^{\xi ia} * ( \sigma^{a}  + \sigma^{qa}_t)$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["gammai"] * self.variables["wia"] * (self.variables["siga"] + self.variables["sigqa"]) ** 2 \
            + (1 - self.variables["gammai"]) * self.variables["sigxia"] * (self.variables["siga"] + self.variables["sigqa"])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula10(self):
        formula_str = r'$w_t^{ia} * \eta_t + w_t^{ha} * (1-\eta_t)$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["wia"] * self.variables["eta"] + self.variables["wha"] * (1 - self.variables["eta"])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula11(self):
        formula_str = r'$c_t^i * \eta_t + c_t^h * (1-\eta_t)$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["ci"] * self.variables["eta"] + self.variables["ch"] * (1 - self.variables["eta"])
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

    def test_latex_formula12(self):
        formula_str = r'$\frac{\rho^i}{1-\frac{1}{\zeta^i}} * \left( \left(\frac{c_t^i}{\xi_t^i} \right)^{1-1/\zeta^i}-1 \right) + \mu^{\xi i} +  \mu^{n i}_t - \frac{\gamma^i}{2} * (\sigma^{nia}_t )^2  - \frac{\gamma^i}{2} * (\sigma_t^{\xi ia})^2 + (1-\gamma^i) * \sigma_t^{\xi ia} * \sigma^{nia}_t$'
        formula = Formula(formula_str, 'eval', self.latex_var_mapping)
        result = formula.eval(self.LOCAL_DICT, self.variables)
        expected_result = self.variables["rhoi"] / (1 - 1 / self.variables["zetai"]) * ((self.variables["ci"] / self.variables["xii"]) ** (1 - 1 / self.variables["zetai"]) - 1) \
        + self.variables["muxii"] + self.variables["muni"] - self.variables["gammai"] / 2 * self.variables["signia"] ** 2 \
        - self.variables["gammai"] / 2 * self.variables["sigxia"] ** 2 \
        + (1 - self.variables["gammai"]) * self.variables["sigxia"] * self.variables["signia"]
        self.assertTrue(torch.allclose(result, expected_result), f"Expected {expected_result}, got {result}")

if __name__ == '__main__':
    unittest.main()

