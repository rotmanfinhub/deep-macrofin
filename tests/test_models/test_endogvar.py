import numpy as np
import torch
import torch.nn as nn

from ecomodels.models import EndogVar


def test_init():
    config = {
        "device": "cpu",
        "hidden_units": [10],
    }
    model = EndogVar("q", ["x", "y", "z"], config)
    assert len(model.model) == 3

def test_init_no_hidden():
    '''
        This should give a single layer neural network    
    '''
    config = {
        "device": "cpu",
        "hidden_units": [],
    }
    model = EndogVar("q", ["x", "y", "z"], config)
    assert len(model.model) == 1

def test_init_positive():
    config = {
        "device": "cpu",
        "hidden_units": [10],
        "positive": True
    }
    model = EndogVar("q", ["x", "y", "z"], config)
    assert len(model.model) == 4 
    
def test_custom_function_computation_identity():
    config = {
        "device": "cpu",
        "test_derivatives": True,
        "hardcode_function": lambda x: x,
    }
    model = EndogVar("q", ["x", "y"], config)
    model.eval()
    x = torch.randn((10, 2), device=model.device) # batch size = 10, var size = 2
    assert torch.allclose(model(x), x)

def test_custom_function_gradients():
    config = {
        "device": "cpu",
        "test_derivatives": True,
        "hardcode_function": lambda x: x[:, 0] * x[:, 1],
    }
    model = EndogVar("qa", ["x", "y"], config)
    model.eval()
    x = torch.randn((10, 2), device=model.device) # batch size = 10, var size = 2
    for (eval_func, expected) in [
        ("qa", x[:, 0] * x[:, 1]),
        ("qa_x", x[:, 1].unsqueeze(-1)),
        ("qa_y", x[:, 0].unsqueeze(-1)),
        ("qa_xx", torch.zeros((10, 1))),
        ("qa_xy", torch.ones((10, 1))),
        ("qa_yx", torch.ones((10, 1))),
        ("qa_yy", torch.zeros((10, 1))),
    ]:
        computed_val = model.derivatives[eval_func](x)
        assert torch.allclose(computed_val, expected), f"Error computing: {eval_func}, expected: {expected}, actual: {computed_val}"

def test_custom_function_gradients_multi_char_input_var():
    config = {
        "device": "cpu",
        "test_derivatives": True,
        "hardcode_function": lambda x: x[:, 0] * x[:, 1],
    }
    model = EndogVar("qa", ["x1", "x2"], config)
    model.eval()
    x = torch.randn((10, 2), device=model.device) # batch size = 10, var size = 2
    for (eval_func, expected) in [
        ("qa", x[:, 0] * x[:, 1]),
        ("qa_x1", x[:, 1].unsqueeze(-1)),
        ("qa_x2", x[:, 0].unsqueeze(-1)),
        ("qa_x1x1", torch.zeros((10, 1))),
        ("qa_x1x2", torch.ones((10, 1))),
        ("qa_x2x1", torch.ones((10, 1))),
        ("qa_x2x2", torch.zeros((10, 1))),
    ]:
        computed_val = model.derivatives[eval_func](x)
        assert torch.allclose(computed_val, expected), f"Error computing: {eval_func}, expected: {expected}, actual: {computed_val}"

def test_random_forward_gradients():
    '''
        This is a single linear layer 3 input 1 output, with all weights = 1,
        so the function should be f(x,y,z) = x+y+z
    '''
    config = {
        "device": "cpu",
        "hidden_units": [],
    }
    model = EndogVar("q", ["x", "y", "z"], config) 
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    model.eval()
    x = torch.randn((10, 3), device=model.device) # batch size = 10, var size = 3
    for (eval_func, expected) in [
        ("q", torch.sum(x, axis=1, keepdim=True)),
        ("q_x", torch.ones((10, 1))),
        ("q_y", torch.ones((10, 1))),
        ("q_z", torch.ones((10, 1))),
        ("q_xx", torch.zeros((10, 1))),
        ("q_xy", torch.zeros((10, 1))),
        ("q_yz", torch.zeros((10, 1))),
        ("q_xyz", torch.zeros((10, 1))),
    ]:
        computed_val = model.derivatives[eval_func](x)
        assert torch.allclose(computed_val, expected), f"Error computing: {eval_func}, expected: {expected}, actual: {computed_val}"

def test_forward_unbatched():
    config = {
        "device": "cpu",
        "hidden_units": [],
    }
    model = EndogVar("q", ["x", "y", "z"], config) 
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    model.eval()
    x = torch.randn(3, device=model.device) # unbatched var size = 3
    for (eval_func, expected) in [
        ("q", torch.sum(x).unsqueeze(0)),
        ("q_x", torch.ones((1, 1))),
        ("q_y", torch.ones((1, 1))),
        ("q_z", torch.ones((1, 1))),
        ("q_xx", torch.zeros((1, 1))),
        ("q_xy", torch.zeros((1, 1))),
        ("q_yz", torch.zeros((1, 1))),
        ("q_xyz", torch.zeros((1, 1))),
    ]:
        computed_val = model.derivatives[eval_func](x)
        assert torch.allclose(computed_val, expected), f"Error computing: {eval_func}, expected: {expected}, actual: {computed_val}"
