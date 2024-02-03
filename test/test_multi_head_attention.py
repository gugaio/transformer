import unittest
import sys
import os
import unittest

# Add the root path to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../transformer')))


import torch
from multi_head_attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.multi_head_attention = MultiHeadAttention(d_model=512, num_heads=8, d_k=64, d_v=64)
        self.X = torch.rand((1, 5, 512))

    def test_forward(self):
        output = self.multi_head_attention.forward(self.X, mask=None)
        self.assertEqual(output[0].shape, (1, 5, 512))


if __name__ == '__main__':
    unittest.main()