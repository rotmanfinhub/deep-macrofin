import unittest

import torch
import torch.nn as nn

from ecomodels.models import Agent, LayerType, LearnableModelType


class TestAgentInitialization(unittest.TestCase):
    def setUp(self):
        self.name = "test_agent"
        self.state_variables = ["x1", "x2", "x3"]

    def test_initialization_with_linear_model(self):
        config = {
            "device": "cpu",
            "hidden_units": [10, 10, 10],
            "positive": False
        }
        agent = Agent(self.name, self.state_variables, config)
        self.assertEqual(agent.model_type, LearnableModelType.Agent)
        self.assertEqual(agent.config["hidden_units"], [10, 10, 10])
        self.assertEqual(agent.config["layer_type"], LayerType.MLP)
        self.assertEqual(agent.device, "cpu")
        self.assertEqual(len(agent.model), 7)  # 3 layers * 2 + 1 output layer

    def test_initialization_with_positive_output(self):
        config = {
            "device": "cpu",
            "hidden_units": [10, 10, 10],
            "positive": True
        }
        agent = Agent(self.name, self.state_variables, config)
        self.assertTrue(agent.config["positive"])

    def test_initialization_with_cuda_device(self):
        config = {
            "hidden_units": [10, 10, 10],
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
            "hidden_units": [],
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

