import unittest

import pytest
import torch

from deep_macrofin import *


class TestPDESystem(unittest.TestCase):
    def setUp(self):
        pde_sys = PDEModel("test")
        pde_sys.set_state(["x", "t"], {"x": [0,1], "t": [0,1]})
        pde_sys.add_agent("q", {"hardcode_function": lambda x: x[:, 0:1] ** 2 + x[:, 1:] ** 3})
        pde_sys.add_endog("psi", {"hardcode_function": lambda x: x[:, 0:1] * x[:, 1:]})
        pde_sys.add_agent("r")
        pde_sys.add_endog_condition("psi", 
                                    "psi(SV)", {"SV": torch.zeros((1, 2))},
                                    Comparator.EQ,
                                    "0", {})  # endogvar_psi_cond_1
        pde_sys.add_agent_condition("q", 
                                    "q(SV)", {"SV": torch.zeros((1, 2))},
                                    Comparator.EQ,
                                    "0", {}) # agent_q_cond_1
        pde_sys.add_equation("y = q + 1") # eq_1
        pde_sys.add_constraint("q", Comparator.GEQ, "0.5") # constraint_1
        pde_sys.add_endog_equation("y = q") # endogeq_1
        pde_sys.add_endog_equation("y = r") # endogeq_2
        pde_sys.add_hjb_equation("q + psi") # hjbeq_1

        all_params = []
        for agent_name, agent in pde_sys.agents.items():
            all_params += list(agent.parameters())
        for endog_var_name, endog_var in pde_sys.endog_vars.items():
            all_params += list(endog_var.parameters())
        
        pde_sys.optimizer = torch.optim.AdamW(all_params, pde_sys.lr)

        self.pde_sys = pde_sys
    
    def run_one_step(self):
        SV = self.pde_sys.sample(0)
        for i, sv_name in enumerate(self.pde_sys.state_variables):
            self.pde_sys.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.pde_sys.test_step(SV)
    
    def test_randomsetup(self):
        # cannot add agent/endog before setting state
        pde_sys = PDEModel("test2")
        with pytest.raises(AssertionError):
            pde_sys.add_agent("q")
        with pytest.raises(AssertionError):
            pde_sys.add_endog("q")

    def test_add_condition_to_non_existing_vars(self):
        with pytest.raises(AssertionError):
            self.pde_sys.add_agent_condition("a", 
                                    "a(SV)", {"SV": torch.zeros((1, 2))},
                                    Comparator.EQ,
                                    "0", {})

        with pytest.raises(AssertionError):
            self.pde_sys.add_endog_condition("a", 
                                    "a(SV)", {"SV": torch.zeros((1, 2))},
                                    Comparator.EQ,
                                    "0", {})

    def test_validation(self):
        self.pde_sys.validate_model_setup()
        self.assertTrue(True)
        with self.assertRaises(Exception):
            self.pde_sys.add_equation("p = a + 1")
            self.pde_sys.validate_model_setup()

    def test_forward_agent(self):
        self.run_one_step()
        sampled_x = self.pde_sys.variable_val_dict["x"]
        sampled_t = self.pde_sys.variable_val_dict["t"]
        expected_q = sampled_x ** 2 + sampled_t ** 3
        result = self.pde_sys.variable_val_dict["q"]
        self.assertTrue(torch.allclose(result, expected_q), f"q: Expected {expected_q}, got {result}")

        expected_q_x = 2 * sampled_x
        result = self.pde_sys.variable_val_dict["q_x"]
        self.assertTrue(torch.allclose(result, expected_q_x), f"q_x: Expected {expected_q_x}, got {result}")

        expected_q_xx = 2 * torch.ones_like(sampled_x)
        result = self.pde_sys.variable_val_dict["q_xx"]
        self.assertTrue(torch.allclose(result, expected_q_xx), f"q_xx: Expected {expected_q_xx}, got {result}")

        expected_q_t = 3 * sampled_t ** 2
        result = self.pde_sys.variable_val_dict["q_t"]
        self.assertTrue(torch.allclose(result, expected_q_t), f"q_t: Expected {expected_q_t}, got {result}")

        expected_q_tt = 6 * sampled_t
        result = self.pde_sys.variable_val_dict["q_tt"]
        self.assertTrue(torch.allclose(result, expected_q_tt), f"q_xx: Expected {expected_q_tt}, got {result}")

    def test_forward_endog(self):
        # perform one step
        self.run_one_step()
        sampled_x = self.pde_sys.variable_val_dict["x"]
        sampled_t = self.pde_sys.variable_val_dict["t"]
        expected_psi = sampled_x * sampled_t
        result = self.pde_sys.variable_val_dict["psi"]
        self.assertTrue(torch.allclose(result, expected_psi), f"psi: Expected {expected_psi}, got {result}")

        expected_psi_x = sampled_t
        result = self.pde_sys.variable_val_dict["psi_x"]
        self.assertTrue(torch.allclose(result, expected_psi_x), f"psi_x: Expected {expected_psi_x}, got {result}")

        expected_psi_xx = torch.zeros_like(sampled_x)
        result = self.pde_sys.variable_val_dict["psi_xx"]
        self.assertTrue(torch.allclose(result, expected_psi_xx), f"psi_xx: Expected {expected_psi_xx}, got {result}")

    def test_forward_endog_cond_loss(self):
        self.run_one_step()
        loss = self.pde_sys.loss_val_dict["endogvar_psi_cond_1"]
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss)), f"endogvar_psi_cond_1: expected: 0, got {loss}")

    def test_forward_agent_cond_loss(self):
        self.run_one_step()
        loss = self.pde_sys.loss_val_dict["agent_q_cond_1"]
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss)), f"agent_q_cond_1: expected: 0, got {loss}")
    
    def test_forward_y(self):
        self.run_one_step()
        sampled_q = self.pde_sys.variable_val_dict["q"]
        expected_y = sampled_q + 1
        result = self.pde_sys.variable_val_dict["y"]
        self.assertTrue(torch.allclose(result, expected_y), f"agent_q_cond_1: expected: {expected_y}, got {result}")

    def test_forward_constraint(self):
        self.run_one_step()
        sampled_q = self.pde_sys.variable_val_dict["q"]
        expected = torch.mean(torch.square(torch.nn.functional.relu(0.5 - sampled_q)))
        result = self.pde_sys.loss_val_dict["constraint_1"]
        self.assertTrue(torch.allclose(result, expected), f"constraint_1: expected: {expected}, got {result}")

    def test_forward_endogeq(self):
        self.run_one_step()
        sampled_q = self.pde_sys.variable_val_dict["q"]
        sampled_y = self.pde_sys.variable_val_dict["y"]
        expected = torch.mean(torch.square(sampled_q - sampled_y))
        result = self.pde_sys.loss_val_dict["endogeq_1"]
        self.assertTrue(torch.allclose(result, expected), f"endogeq_1: expected: {expected}, got {result}")

    def test_forward_endogeq2(self):
        self.run_one_step()
        sampled_y = self.pde_sys.variable_val_dict["y"]
        sampled_r = self.pde_sys.variable_val_dict["r"]
        expected = torch.mean(torch.square(sampled_y - sampled_r))
        result = self.pde_sys.loss_val_dict["endogeq_2"]
        self.assertTrue(torch.allclose(result, expected), f"endogeq_2: expected: {expected}, got {result}")

    def test_forward_hjb(self):
        self.run_one_step()
        sampled_q = self.pde_sys.variable_val_dict["q"]
        sampled_psi = self.pde_sys.variable_val_dict["psi"]
        expected = torch.mean(torch.square(sampled_q + sampled_psi))
        result = self.pde_sys.loss_val_dict["hjbeq_1"]
        self.assertTrue(torch.allclose(result, expected), f"hjbeq_1: expected: {expected}, got {result}")

if __name__ == "__main__":
    unittest.main()