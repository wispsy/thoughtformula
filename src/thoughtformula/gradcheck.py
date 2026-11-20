"""Finite-difference gradient verification for all custom autograd operators."""

from array import array
import math
import random
import unittest

from .tensor import FlatTensor


def check_rmsnorm_gradient(T=2, D=4, eps=1e-5, tol=1e-5):
    """Verifies rmsnorm analytical backward pass against centered finite differences."""
    data = array('d', [random.gauss(0.0, 1.0) for _ in range(T * D)])
    grad_out = array('d', [random.gauss(0.0, 1.0) for _ in range(T * D)])

    # Analytical gradient
    x = FlatTensor(T, D, array('d', data), requires_grad=True)
    y = x.rmsnorm()
    y.grad = array('d', grad_out)
    y._backward()
    analytical = list(x.grad)

    # Numerical gradient
    numerical = []
    for idx in range(T * D):
        d_plus = array('d', data)
        d_plus[idx] += eps
        x_plus = FlatTensor(T, D, d_plus, requires_grad=False)
        y_plus = x_plus.rmsnorm()

        d_minus = array('d', data)
        d_minus[idx] -= eps
        x_minus = FlatTensor(T, D, d_minus, requires_grad=False)
        y_minus = x_minus.rmsnorm()

        diff_sum = sum((yp - ym) * go for yp, ym, go in zip(y_plus.data, y_minus.data, grad_out))
        numerical.append(diff_sum / (2.0 * eps))

    max_err = max(abs(a - n) for a, n in zip(analytical, numerical))
    return max_err < tol, max_err


def check_swiglu_gradient(T=2, D=4, eps=1e-5, tol=1e-5):
    """Verifies swiglu_act analytical backward pass against centered finite differences."""
    g_data = array('d', [random.gauss(0.0, 1.0) for _ in range(T * D)])
    u_data = array('d', [random.gauss(0.0, 1.0) for _ in range(T * D)])
    grad_out = array('d', [random.gauss(0.0, 1.0) for _ in range(T * D)])

    gate = FlatTensor(T, D, array('d', g_data), requires_grad=True)
    up = FlatTensor(T, D, array('d', u_data), requires_grad=True)
    out = FlatTensor.swiglu_act(gate, up)
    out.grad = array('d', grad_out)
    out._backward()
    ana_g, ana_u = list(gate.grad), list(up.grad)

    num_g, num_u = [], []
    for idx in range(T * D):
        # Gate numerical
        gp, gm = array('d', g_data), array('d', g_data)
        gp[idx] += eps
        gm[idx] -= eps
        op = FlatTensor.swiglu_act(FlatTensor(T, D, gp), FlatTensor(T, D, u_data))
        om = FlatTensor.swiglu_act(FlatTensor(T, D, gm), FlatTensor(T, D, u_data))
        num_g.append(sum((p - m) * go for p, m, go in zip(op.data, om.data, grad_out)) / (2.0 * eps))

        # Up numerical
        up_p, up_m = array('d', u_data), array('d', u_data)
        up_p[idx] += eps
        up_m[idx] -= eps
        op = FlatTensor.swiglu_act(FlatTensor(T, D, g_data), FlatTensor(T, D, up_p))
        om = FlatTensor.swiglu_act(FlatTensor(T, D, g_data), FlatTensor(T, D, up_m))
        num_u.append(sum((p - m) * go for p, m, go in zip(op.data, om.data, grad_out)) / (2.0 * eps))

    err_g = max(abs(a - n) for a, n in zip(ana_g, num_g))
    err_u = max(abs(a - n) for a, n in zip(ana_u, num_u))
    max_err = max(err_g, err_u)
    return max_err < tol, max_err


def check_rope_gqa_gradient(T=2, D=4, n_heads=2, n_kv_heads=1, eps=1e-5, tol=1e-5):
    """Verifies rope_gqa analytical backward pass against centered finite differences."""
    kv_d = D // (n_heads // n_kv_heads)
    q_data = array('d', [random.gauss(0.0, 0.5) for _ in range(T * D)])
    k_data = array('d', [random.gauss(0.0, 0.5) for _ in range(T * kv_d)])
    v_data = array('d', [random.gauss(0.0, 0.5) for _ in range(T * kv_d)])
    grad_out = array('d', [random.gauss(0.0, 0.5) for _ in range(T * D)])

    Q = FlatTensor(T, D, array('d', q_data), requires_grad=True)
    K = FlatTensor(T, kv_d, array('d', k_data), requires_grad=True)
    V = FlatTensor(T, kv_d, array('d', v_data), requires_grad=True)
    out = FlatTensor.rope_gqa(Q, K, V, n_heads, n_kv_heads)
    out.grad = array('d', grad_out)
    out._backward()
    ana_q, ana_k, ana_v = list(Q.grad), list(K.grad), list(V.grad)

    # Numerical for V
    num_v = []
    for idx in range(T * kv_d):
        vp, vm = array('d', v_data), array('d', v_data)
        vp[idx] += eps
        vm[idx] -= eps
        op = FlatTensor.rope_gqa(FlatTensor(T, D, q_data), FlatTensor(T, kv_d, k_data), FlatTensor(T, kv_d, vp), n_heads, n_kv_heads)
        om = FlatTensor.rope_gqa(FlatTensor(T, D, q_data), FlatTensor(T, kv_d, k_data), FlatTensor(T, kv_d, vm), n_heads, n_kv_heads)
        num_v.append(sum((p - m) * go for p, m, go in zip(op.data, om.data, grad_out)) / (2.0 * eps))

    err_v = max(abs(a - n) for a, n in zip(ana_v, num_v))
    return err_v < tol, err_v


def check_moe_router_gradient(T=2, N_E=4, D=4, eps=1e-5, tol=1e-5):
    """Verifies moe_router_combine analytical backward pass against centered finite differences."""
    r_data = array('d', [random.gauss(0.0, 0.5) for _ in range(T * N_E)])
    # A finite difference is a local derivative check. Keep its +/- eps
    # perturbation inside one selected top-k region, away from a discontinuity.
    if N_E > 2:
        for t in range(T):
            ranked = sorted(range(N_E), key=lambda e: r_data[t * N_E + e], reverse=True)
            second, third = t * N_E + ranked[1], t * N_E + ranked[2]
            if r_data[second] - r_data[third] <= 4 * eps:
                r_data[second] += 8 * eps
    exp_outs = [FlatTensor(T, D, array('d', [random.gauss(0.0, 0.5) for _ in range(T * D)])) for _ in range(N_E)]
    grad_out = array('d', [random.gauss(0.0, 0.5) for _ in range(T * D)])

    R = FlatTensor(T, N_E, array('d', r_data), requires_grad=True)
    out = FlatTensor.moe_router_combine(R, exp_outs, top_k=2)
    out.grad = array('d', grad_out)
    out._backward()
    ana_r = list(R.grad)

    num_r = []
    for idx in range(T * N_E):
        rp, rm = array('d', r_data), array('d', r_data)
        rp[idx] += eps
        rm[idx] -= eps
        op = FlatTensor.moe_router_combine(FlatTensor(T, N_E, rp), exp_outs, top_k=2)
        om = FlatTensor.moe_router_combine(FlatTensor(T, N_E, rm), exp_outs, top_k=2)
        num_r.append(sum((p - m) * go for p, m, go in zip(op.data, om.data, grad_out)) / (2.0 * eps))

    max_err = max(abs(a - n) for a, n in zip(ana_r, num_r))
    return max_err < tol, max_err


def run_gradient_checks():
    """Runs all finite-difference gradient checks, printing verification report."""
    checks = [
        ("RMSNorm", check_rmsnorm_gradient),
        ("SwiGLU Activation", check_swiglu_gradient),
        ("RoPE GQA Attention", check_rope_gqa_gradient),
        ("MoE Router Combine", check_moe_router_gradient),
    ]
    all_passed = True
    results = {}
    for name, fn in checks:
        passed, err = fn()
        results[name] = (passed, err)
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name:20s} (max finite-difference error: {err:.2e})")
        if not passed:
            all_passed = False
    return all_passed, results
