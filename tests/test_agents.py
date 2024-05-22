import torch
import torch.nn as nn
from ecomodels.models.agents import Agent  # assuming the refactored Agent class is in agent.py
import unittest


class TestAgentInitialization(unittest.TestCase):
    def setUp(self):
        self.name = "test_agent"
        self.state_variables = ["x1", "x2", "x3"]

    def test_initialization_with_linear_model(self):
        config = {
            "output_size": 10,
            "num_layers": 3,
            "model_type": "linear",
            "positive": False,
            "test_derivatives": False,
            "device": "cpu"
        }
        agent = Agent(self.name, self.state_variables, config)
        self.assertEqual(agent.output_size, 10)
        self.assertEqual(agent.num_layers, 3)
        self.assertEqual(agent.model_type, "linear")
        self.assertFalse(agent.positive)
        self.assertFalse(agent.test_derivatives)
        self.assertEqual(agent.device, "cpu")
        self.assertEqual(len(agent.net), 7)  # 3 layers * 2 + 1 output layer

    def test_initialization_with_positive_output(self):
        config = {
            "output_size": 5,
            "num_layers": 2,
            "model_type": "active learning",
            "positive": True,
            "test_derivatives": False,
            "device": "cpu"
        }
        agent = Agent(self.name, self.state_variables, config)
        self.assertTrue(agent.positive)

    def test_initialization_with_cuda_device(self):
        config = {
            "output_size": 10,
            "num_layers": 3,
            "model_type": "linear",
            "positive": False,
            "test_derivatives": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        agent = Agent(self.name, self.state_variables, config)
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(agent.device, expected_device)


class TestAgentDerivatives(unittest.TestCase):
    def setUp(self):
        self.name = "test_agent"
        self.state_variables = ["x1", "x2", "x3"]
        self.config = {
            "output_size": 1,
            "num_layers": 1,
            "model_type": "linear",
            "positive": False,
            "test_derivatives": True,
            "device": "cpu"
        }
        self.agent = Agent(self.name, self.state_variables, self.config)

    def test_derivative_computation(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        output = self.agent(x)
        output.backward()
        # Here, you would need to test the derivative values. This is an example placeholder.
        # Check the gradients in x
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

