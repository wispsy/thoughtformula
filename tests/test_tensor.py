"""Unit tests for FlatTensor autograd, quantization, and ops."""

from array import array
import math
import unittest

from thoughtformula.tensor import FlatTensor


class TestFlatTensor(unittest.TestCase):
    def test_init_and_properties(self):
        t = FlatTensor(2, 3)
        self.assertEqual(t.rows, 2)
        self.assertEqual(t.cols, 3)
        self.assertEqual(len(t.data), 6)
        self.assertEqual(len(t.grad), 6)
        self.assertTrue(t.requires_grad)

    def test_quantize_int8_and_roundtrip(self):
        t = FlatTensor(2, 2, array('d', [1.0, -1.0, 0.5, 0.0]))
        t.quantize_int8()
        self.assertIsNotNone(t.q_data)
        self.assertEqual(len(t.q_data), 4)
        self.assertAlmostEqual(t.q_scale, 1.0 / 127.0, places=5)
        self.assertEqual(t.q_data[0], 127)
        self.assertEqual(t.q_data[1], -127)

        d = t.to_int8_compressed_dict()
        self.assertEqual(d["rows"], 2)
        self.assertEqual(d["cols"], 2)

        restored = FlatTensor.from_serialized(d)
        self.assertEqual(restored.rows, 2)
        self.assertEqual(restored.cols, 2)
        self.assertAlmostEqual(restored.data[0], 1.0, places=2)
        self.assertAlmostEqual(restored.data[1], -1.0, places=2)

    def test_add_and_backward(self):
        a = FlatTensor(2, 2, array('d', [1.0, 2.0, 3.0, 4.0]))
        b = FlatTensor(2, 2, array('d', [5.0, 6.0, 7.0, 8.0]))
        c = a + b
        self.assertEqual(list(c.data), [6.0, 8.0, 10.0, 12.0])

        c.grad = array('d', [1.0, 1.0, 1.0, 1.0])
        c._backward()
        self.assertEqual(list(a.grad), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(list(b.grad), [1.0, 1.0, 1.0, 1.0])

    def test_add_broadcast_and_backward(self):
        a = FlatTensor(2, 2, array('d', [1.0, 2.0, 3.0, 4.0]))
        b = FlatTensor(1, 2, array('d', [10.0, 20.0]))
        c = a + b
        self.assertEqual(list(c.data), [11.0, 22.0, 13.0, 24.0])

        c.grad = array('d', [1.0, 1.0, 1.0, 1.0])
        c._backward()
        self.assertEqual(list(a.grad), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(list(b.grad), [2.0, 2.0])

    def test_scale_and_backward(self):
        a = FlatTensor(1, 3, array('d', [2.0, -3.0, 4.0]))
        b = a.scale(2.5)
        self.assertAlmostEqual(b.data[0], 5.0)
        self.assertAlmostEqual(b.data[1], -7.5)
        self.assertAlmostEqual(b.data[2], 10.0)

        b.grad = array('d', [1.0, 1.0, 1.0])
        b._backward()
        self.assertAlmostEqual(a.grad[0], 2.5)
        self.assertAlmostEqual(a.grad[1], 2.5)
        self.assertAlmostEqual(a.grad[2], 2.5)

    def test_matmul_forward_and_backward(self):
        # A: [2, 3], B: [3, 2]
        A = FlatTensor(2, 3, array('d', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        B = FlatTensor(3, 2, array('d', [7.0, 8.0, 9.0, 1.0, 2.0, 3.0]))
        C = A.matmul(B)
        self.assertEqual(C.rows, 2)
        self.assertEqual(C.cols, 2)
        # Row 0: 1*7 + 2*9 + 3*2 = 7+18+6 = 31, 1*8 + 2*1 + 3*3 = 8+2+9 = 19
        # Row 1: 4*7 + 5*9 + 6*2 = 28+45+12 = 85, 4*8 + 5*1 + 6*3 = 32+5+18 = 55
        self.assertAlmostEqual(C.data[0], 31.0)
        self.assertAlmostEqual(C.data[1], 19.0)
        self.assertAlmostEqual(C.data[2], 85.0)
        self.assertAlmostEqual(C.data[3], 55.0)

        C.grad = array('d', [1.0, 0.0, 0.0, 1.0])
        C._backward()
        # dA = dOut @ B.T -> [[1, 0], [0, 1]] @ [[7, 9, 2], [8, 1, 3]]
        # Row 0: [7, 9, 2], Row 1: [8, 1, 3]
        self.assertAlmostEqual(A.grad[0], 7.0)
        self.assertAlmostEqual(A.grad[1], 9.0)
        self.assertAlmostEqual(A.grad[2], 2.0)
        self.assertAlmostEqual(A.grad[3], 8.0)
        self.assertAlmostEqual(A.grad[4], 1.0)
        self.assertAlmostEqual(A.grad[5], 3.0)

    def test_rmsnorm(self):
        X = FlatTensor(1, 4, array('d', [1.0, 1.0, 1.0, 1.0]))
        Y = X.rmsnorm()
        # RMS = sqrt(4 / 4) = 1.0
        for val in Y.data:
            self.assertAlmostEqual(val, 1.0, places=4)

    def test_swiglu_act(self):
        gate = FlatTensor(1, 2, array('d', [0.0, 2.0]))
        up = FlatTensor(1, 2, array('d', [3.0, 4.0]))
        out = FlatTensor.swiglu_act(gate, up)
        # sigmoid(0) = 0.5 -> 0.0 * 0.5 * 3.0 = 0.0
        self.assertAlmostEqual(out.data[0], 0.0)
        # sigmoid(2) = 1 / (1 + e^-2) ~ 0.8808 -> 2.0 * 0.8808 * 4.0 ~ 7.046
        self.assertTrue(out.data[1] > 6.0)

    def test_transpose(self):
        A = FlatTensor(2, 3, array('d', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        A.quantize_int8()
        AT = A.transpose()
        self.assertEqual(AT.rows, 3)
        self.assertEqual(AT.cols, 2)
        self.assertEqual(list(AT.data), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        self.assertIsNotNone(AT.q_data)
        self.assertEqual(len(AT.q_data), 6)

    def test_matmul_sumprod_backward_both_grads(self):
        A = FlatTensor(2, 2, array('d', [1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        B = FlatTensor(2, 2, array('d', [0.5, -0.5, 1.0, 2.0]), requires_grad=True)
        C = A.matmul(B)
        C.backward()
        # When C.backward() is called on [2, 2], grad[0] = 1.0, grad[1:] = 0.0
        # dOut = [[1.0, 0.0], [0.0, 0.0]]
        # dA = dOut @ B^T = [[0.5, 1.0], [0.0, 0.0]]
        # dB = A^T @ dOut = [[1.0, 0.0], [2.0, 0.0]]
        self.assertAlmostEqual(A.grad[0], 0.5)
        self.assertAlmostEqual(A.grad[1], 1.0)
        self.assertAlmostEqual(B.grad[0], 1.0)
        self.assertAlmostEqual(B.grad[2], 2.0)

    def test_cross_entropy(self):
        logits = FlatTensor(2, 3, array('d', [10.0, 0.0, 0.0, 0.0, 10.0, 0.0]))
        loss = FlatTensor.cross_entropy(logits, [0, 1])
        # Very high confidence on correct classes -> loss close to 0
        self.assertTrue(loss.data[0] < 0.01)
        loss.backward()
        self.assertTrue(abs(logits.grad[0]) < 0.01)


if __name__ == "__main__":
    unittest.main()
