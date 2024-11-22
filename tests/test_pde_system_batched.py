import unittest

import pytest
import torch

from deep_macrofin import *


class TestPDESystem(unittest.TestCase):
    def setUp(self):
        pde_sys = PDEModel("test")
        pde_sys.set_state(["x", "t"], {"x": [0,1], "t": [0,1]})
        pde_sys.add_agent("q", {"hardcode_function": lambda x: x[..., 0:1] ** 2 + x[..., 1:] ** 3, "batch_jac_hes": True})
        pde_sys.add_endog("psi", {"hardcode_function": lambda x: x[..., 0:1] * x[..., 1:], "batch_jac_hes": True})
        pde_sys.add_endog("k", {"hardcode_function": lambda x: x[..., 0:1] ** 2 + x[..., 1:] ** 2, "batch_jac_hes": True})
        pde_sys.add_agent("xi")
        pde_sys.add_equation("y = torch.einsum('boi->bo', psi_Jac)")
        pde_sys.add_equation("q_x = q_Jac[:,0,:1]")
        pde_sys.add_equation("q_t = q_Jac[:,0,1:]")
        pde_sys.add_equation("q_xx = q_Hess[:, 0, :1, :1]")
        pde_sys.add_equation("q_yx = q_Hess[:, 0, 1:, :1]")
        pde_sys.add_equation("k_lap = torch.diagonal(k_Hess, dim1=2, dim2=3)")

        all_params = []
        for agent_name, agent in pde_sys.agents.items():
            all_params += list(agent.parameters())
        for endog_var_name, endog_var in pde_sys.endog_vars.items():
            all_params += list(endog_var.parameters())
        
        pde_sys.optimizer = torch.optim.AdamW(all_params, pde_sys.lr)

        self.pde_sys = pde_sys
    
    def run_one_step(self):
        SV = self.pde_sys.sample(0)
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.pde_sys.state_variables):
            self.pde_sys.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.pde_sys.variable_val_dict["SV"] = SV
        self.pde_sys.test_step(SV)

    def test_jacobian(self):
        self.run_one_step()
        sampled_x = self.pde_sys.variable_val_dict["x"]
        sampled_t = self.pde_sys.variable_val_dict["t"]

        expected_y = sampled_x + sampled_t
        result = self.pde_sys.variable_val_dict["y"]
        self.assertTrue(torch.allclose(result, expected_y), f"y: Expected {expected_y}, got {result}")

    def test_first_order(self):
        self.run_one_step()
        sampled_x = self.pde_sys.variable_val_dict["x"]
        sampled_t = self.pde_sys.variable_val_dict["t"]

        expected_qx = 2 * sampled_x
        result = self.pde_sys.variable_val_dict["q_x"]
        self.assertTrue(torch.allclose(result, expected_qx), f"q_x: Expected {expected_qx}, got {result}")

        expected_qt = 3 * sampled_t ** 2
        result = self.pde_sys.variable_val_dict["q_t"]
        self.assertTrue(torch.allclose(result, expected_qt), f"q_t: Expected {expected_qt}, got {result}")

    def test_second_order(self):
        self.run_one_step()
        expected_qxx = torch.ones((self.pde_sys.config["batch_size"], 1)) * 2
        result = self.pde_sys.variable_val_dict["q_xx"]
        self.assertTrue(torch.allclose(result, expected_qxx), f"q_x: Expected {expected_qxx}, got {result}")

        expected_qyx = torch.zeros((self.pde_sys.config["batch_size"], 1))
        result = self.pde_sys.variable_val_dict["q_yx"]
        self.assertTrue(torch.allclose(result, expected_qyx), f"q_x: Expected {expected_qxx}, got {result}")

    def test_laplacian(self):
        self.run_one_step()
        expected_k_lap = torch.ones((self.pde_sys.config["batch_size"], 2, 2)) * 2
        result = self.pde_sys.variable_val_dict["k_lap"]
        self.assertTrue(torch.allclose(result, expected_k_lap), f"q_x: Expected {expected_k_lap}, got {result}")

if __name__ == "__main__":
    unittest.main()