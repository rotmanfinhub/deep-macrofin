from deepMacroFin.models import *


if __name__ == "__main__":
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