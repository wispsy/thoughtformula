"""Unit tests for finite-difference gradient verification."""

import unittest
from thoughtformula.gradcheck import (
    check_rmsnorm_gradient,
    check_swiglu_gradient,
    check_rope_gqa_gradient,
    check_moe_router_gradient,
    run_gradient_checks,
)


class TestGradCheck(unittest.TestCase):
    def test_rmsnorm_grad(self):
        passed, err = check_rmsnorm_gradient()
        self.assertTrue(passed)
        self.assertTrue(err < 1e-5)

    def test_swiglu_grad(self):
        passed, err = check_swiglu_gradient()
        self.assertTrue(passed)
        self.assertTrue(err < 1e-5)

    def test_rope_gqa_grad(self):
        passed, err = check_rope_gqa_gradient()
        self.assertTrue(passed)
        self.assertTrue(err < 1e-5)

    def test_moe_router_grad(self):
        passed, err = check_moe_router_gradient()
        self.assertTrue(passed)
        self.assertTrue(err < 1e-5)

    def test_run_all_checks(self):
        all_passed, results = run_gradient_checks()
        self.assertTrue(all_passed)
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
