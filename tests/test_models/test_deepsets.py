import unittest

import torch
import torch.nn as nn

from deep_macrofin.models import DeepSet, ActivationType
from torch.func import vmap, jacrev

class TestDeepSets(unittest.TestCase):

    def init_model(self, input_size_sym, input_size_ext, output_size):
        return DeepSet({
            "input_size_sym": input_size_sym,
            "input_size_ext": input_size_ext,
            "output_size": output_size,
            "activation_type": ActivationType.Tanh,
        })
    
    def compute_res(self, net, x):
        x_swapped = x.clone()
        x_swapped[:, (0, 1)] = x_swapped[:, (1, 0)] # swap first two columns, output should be unchanged
        
        y1 = net.forward(x)
        y2 = vmap(net.forward)(x)
        y2_x = vmap(jacrev(net.forward))(x)
        y3 = net.forward(x_swapped)
        y3_swapped = y3.clone()
        y3_swapped[:, (0, 1)] = y3_swapped[:, (1, 0)] # swap back to compare

        return y1, y2, y2_x, y3, y3_swapped

    def _run_basic_test(self, input_size_sym, input_size_ext, output_size):
        net = self.init_model(input_size_sym, input_size_ext, output_size)
        x = torch.rand((10, input_size_sym + input_size_ext))
        y1, y2, y2_x, y3, y3_swapped = self.compute_res(net, x)

        self.assertTrue(torch.allclose(y1, y2, atol=1e-5), "vmap output doesn't match regular forward")
        self.assertTrue(torch.allclose(y1, y3_swapped, atol=1e-5), "Equivariance test failed")
        self.assertEqual(y2_x.shape, (10, output_size, input_size_sym + input_size_ext))

    def test_deepset_small_sym(self):
        self._run_basic_test(input_size_sym=5, input_size_ext=0, output_size=5)

    def test_deepset_larger_sym(self):
        self._run_basic_test(input_size_sym=5, input_size_ext=0, output_size=7)

    def test_deepset_with_ext(self):
        self._run_basic_test(input_size_sym=5, input_size_ext=2, output_size=7)


if __name__ == "__main__":
    unittest.main()