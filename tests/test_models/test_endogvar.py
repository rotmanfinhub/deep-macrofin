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
        "hardcode_function": lambda x: x,
    }
    model = EndogVar("q", ["x", "y"], config)
    model.eval()
    x = torch.randn((10, 2), device=model.device) # batch size = 10, var size = 2
    assert torch.allclose(model(x), x)

def test_custom_function_gradients():
    config = {
        "device": "cpu",
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

def test_custom_function_gradients2():
    config = {
        "device": "cpu",
        "hardcode_function": lambda x: x[:, 0] ** 2 * x[:, 1],
    }
    model = EndogVar("qa", ["x", "y"], config)
    model.eval()
    x = torch.randn((10, 2), device=model.device) # batch size = 10, var size = 2
    for (eval_func, expected) in [
        ("qa", x[:, 0] ** 2 * x[:, 1]),
        ("qa_x", 2 * x[:, 0:1] * x[:, 1:2]),
        ("qa_y", x[:, 0:1] ** 2),
        ("qa_xx", 2 * x[:, 1:2]),
        ("qa_xy", 2 * x[:, 0:1]),
        ("qa_yx", 2 * x[:, 0:1]),
        ("qa_yy", torch.zeros((10, 1))),
    ]:
        computed_val = model.derivatives[eval_func](x)
        assert torch.allclose(computed_val, expected), f"Error computing: {eval_func}, expected: {expected}, actual: {computed_val}"

def test_custom_function_gradients3():
    config = {
        "device": "cpu",
        "hardcode_function": lambda x: torch.sin(x[:, 0]) * torch.cos(x[:, 1]),
    }
    model = EndogVar("qa", ["x", "y"], config)
    model.eval()
    x = torch.randn((10, 2), device=model.device) # batch size = 10, var size = 2
    for (eval_func, expected) in [
        ("qa", torch.sin(x[:, 0]) * torch.cos(x[:, 1])),
        ("qa_x", torch.cos(x[:, 0:1]) * torch.cos(x[:, 1:2])),
        ("qa_y", -torch.sin(x[:, 0:1]) * torch.sin(x[:, 1:2])),
        ("qa_xx", -torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])),
        ("qa_xy", -torch.cos(x[:, 0:1]) * torch.sin(x[:, 1:2])),
        ("qa_yx", -torch.cos(x[:, 0:1]) * torch.sin(x[:, 1:2])),
        ("qa_yy", -torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])),
    ]:
        computed_val = model.derivatives[eval_func](x)
        assert torch.allclose(computed_val, expected), f"Error computing: {eval_func}, expected: {expected}, actual: {computed_val}"

def test_custom_function_gradients_multi_char_input_var():
    config = {
        "device": "cpu",
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

def test_custom_function_high_order():
    config = {
        "device": "cpu",
        "hardcode_function": lambda x: x ** 4,
        "min_derivative_order": 6,
    }
    model = EndogVar("qa", ["x"], config)
    model.eval()
    x = torch.randn((10, 1), device=model.device) # batch size = 10, var size = 2
    for (eval_func, expected) in [
        ("qa", x ** 4),
        ("qa_x", 4 * x ** 3),
        ("qa_xx", 12 * x ** 2),
        ("qa_xxx", 24 * x),
        ("qa_xxxx", 24 * torch.ones_like(x)),
        ("qa_xxxxx", torch.zeros_like(x)),
        ("qa_xxxxxx", torch.zeros_like(x)),
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
