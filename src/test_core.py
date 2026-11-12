"""Kiem thu loi thuat toan theo thu tu tang dan (Buoc 4 cua de bai).
Chay:  python test_core.py     (hoac: pytest test_core.py)
"""
import os
import numpy as np
from gd_linreg import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "winequality-red.csv")


def hand_example():
    """4 diem (x,y): (0,1) (1,3) (2,4) (3,8). Giai tay phuong trinh chuan tac:
        N=4, sum x=6, sum x^2=14, sum y=16, sum xy=35
        [4 6; 6 14] [b; w] = [16; 35],  det = 20
        b = (14*16 - 6*35)/20 = 0.7,   w = (4*35 - 6*16)/20 = 2.2
    """
    x = np.array([0., 1., 2., 3.])
    y = np.array([1., 3., 4., 8.])
    X = add_bias(x[:, None])
    return X, y, np.array([0.7, 2.2])


def test_1_hand_example_normal_equation():
    X, y, th_hand = hand_example()
    assert np.allclose(normal_equation(X, y), th_hand, atol=1e-12)


def test_1_hand_example_gradient_descent():
    X, y, th_hand = hand_example()
    lmax, lmin, k = hessian_spectrum(X)
    th, _ = gradient_descent(X, y, 1 / lmax, T=5000)
    assert np.allclose(th, th_hand, atol=1e-8)
    # buoc hoc > 2/lmax phai phan ky
    _, h = gradient_descent(X, y, 2.05 / lmax, T=5000)
    assert h["diverged"]
    # heavy-ball
    g, b = heavy_ball_params(lmax, lmin)
    th2, _ = gradient_descent(X, y, g, b, T=2000)
    assert np.allclose(th2, th_hand, atol=1e-10)


def test_2_gradient_check():
    X, y, _ = make_synthetic()
    assert gradient_check(X, y) < 1e-6


def test_2_synthetic_recovery():
    X, y, th_true = make_synthetic()
    th_ne = normal_equation(X, y)
    lmax = hessian_spectrum(X)[0]
    th_gd, _ = gradient_descent(X, y, 1 / lmax, T=20000)
    assert np.abs(th_gd - th_ne).max() < 1e-10          # GD = nghiem chuan tac
    assert np.abs(th_ne - th_true).max() < 10 * 0.05 / np.sqrt(500)  # sai so thong ke


def test_3_split_and_standardize():
    X, y, _ = load_wine(DATA)
    assert X.shape == (1599, 11) and y.shape == (1599,)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, SEED)
    assert len(ytr) == 1279 and len(yte) == 320
    m, s = standardize_fit(Xtr)
    Z = standardize_apply(Xtr, m, s)
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1)
    # H chuan hoa: tr(H) = D+1
    H = hessian(add_bias(Z))
    assert np.isclose(np.trace(H), 12)
    assert np.linalg.matrix_rank(add_bias(Z)) == 12      # Menh de 1: du hang cot


def test_3_real_data_matches_normal_equation():
    X, y, _ = load_wine(DATA)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, SEED)
    m, s = standardize_fit(Xtr)
    Z = add_bias(standardize_apply(Xtr, m, s))
    th = normal_equation(Z, ytr)
    lmax = hessian_spectrum(Z)[0]
    t, _ = gradient_descent(Z, ytr, 1 / lmax, T=2000)
    assert np.abs(t - th).max() < 1e-12


def test_heavy_ball_theory():
    lmax, lmin = 3.0, 0.06
    g, b = heavy_ball_params(lmax, lmin)
    # moi lam trong [lmin, lmax]: mo-dun nghiem = sqrt(beta)
    for lam in np.linspace(lmin, lmax, 50):
        assert np.isclose(heavy_ball_root_modulus(g, b, lam), np.sqrt(b), atol=1e-6)
    assert g < heavy_ball_gamma_limit(lmax, b)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        f()
        print("PASS", n)
    print(f"\n{len(tests)}/{len(tests)} kiem thu dat.")
