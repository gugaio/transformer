import unittest
import sys
import os
import unittest

# Add the root path to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../transformer')))


import torch
from attention import Attention

class TestAttention(unittest.TestCase):
    def setUp(self):
        self.attention = Attention()
        self.q = torch.rand(2, 3, 4, 5)
        self.k = torch.rand(2, 3, 4, 5)
        self.v = torch.rand(2, 3, 4, 5)
        self.mask = torch.zeros(2, 1, 1, 4)

    def test_forward(self):
        output, attn = self.attention(self.q, self.k, self.v, self.mask)
        self.assertEqual(output.shape, (2, 3, 4, 5))
        self.assertEqual(attn.shape, (2, 3, 4, 4))

if __name__ == '__main__':
    unittest.main()