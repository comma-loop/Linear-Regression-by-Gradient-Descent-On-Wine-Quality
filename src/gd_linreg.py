"""Loi thuat toan: hoi quy tuyen tinh bang ha gradient (chi dung numpy)."""
import numpy as np


def load_wine(path):
    d = np.genfromtxt(path, delimiter=";", skip_header=1)
    with open(path) as f:
        names = [s.strip('"') for s in f.readline().strip().split(";")]
    return d[:, :-1], d[:, -1], names[:-1]


def train_test_split(X, y, test_size=0.2, seed=42):
    """Hoan vi ngau nhien voi rng(seed); lay `test_size` dau lam tap kiem tra."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(round(test_size * len(y)))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def standardize_fit(X):
    return X.mean(axis=0), X.std(axis=0)


def standardize_apply(X, m, s):
    return (X - m) / s


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def loss(theta, X, y):
    r = X @ theta - y
    return 0.5 * np.mean(r ** 2)          # L = 1/(2N) ||y - X theta||^2


def grad(theta, X, y):
    return X.T @ (X @ theta - y) / X.shape[0]   # (3)


def hessian(X):
    return X.T @ X / X.shape[0]


def hessian_spectrum(X):
    ev = np.linalg.eigvalsh(hessian(X))
    return ev.max(), ev.min(), ev.max() / ev.min()


def normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)      # (4)


def gradient_descent(X, y, gamma, beta=0.0, T=5000, eps=0.0,
                     theta_star=None, tol=None):
    """Thuat toan 1. beta=0: GD thuong. Tra ve theta va lich su."""
    N, P = X.shape
    theta = np.zeros(P)
    v = np.zeros(P)
    hist = {"loss": [loss(theta, X, y)], "dist": [], "iters_to_tol": None,
            "diverged": False}
    if theta_star is not None:
        hist["dist"].append(np.linalg.norm(theta - theta_star))
    for t in range(1, T + 1):
        g = grad(theta, X, y)
        if np.linalg.norm(g) < eps:
            break
        v = beta * v - gamma * g
        theta = theta + v
        L = loss(theta, X, y)
        hist["loss"].append(L)
        if theta_star is not None:
            d = np.linalg.norm(theta - theta_star)
            hist["dist"].append(d)
            if tol is not None and hist["iters_to_tol"] is None and d < tol:
                hist["iters_to_tol"] = t
        if not np.isfinite(L) or L > 1e30:
            hist["diverged"] = True
            break
    return theta, hist


def heavy_ball_params(lmax, lmin):
    kappa = lmax / lmin
    gamma = 4.0 / (np.sqrt(lmax) + np.sqrt(lmin)) ** 2
    beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    return gamma, beta


def mse(y, yhat):
    return np.mean((y - yhat) ** 2)


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------------------
# Cac ham bo tro cho kiem thu va phan tich (van chi dung numpy)
# ---------------------------------------------------------------------------
SEED = 42            # chia du lieu + du lieu tong hop
SEED_GRADCHECK = 0   # theta ngau nhien cho kiem tra gradient
EPS_FD = 1e-5        # buoc sai phan huu han trung tam


def gradient_check(X, y, seed=SEED_GRADCHECK, eps=EPS_FD):
    """Sai khac lon nhat giua grad giai tich (3) va sai phan huu han trung tam.
    theta ~ N(0, I) sinh boi default_rng(seed)."""
    th = np.random.default_rng(seed).normal(size=X.shape[1])
    I = np.eye(len(th))
    fd = np.array([(loss(th + eps * I[i], X, y) - loss(th - eps * I[i], X, y)) / (2 * eps)
                   for i in range(len(th))])
    return np.abs(fd - grad(th, X, y)).max()


def make_synthetic(seed=SEED, N=500, D=11, sigma=0.05):
    """Du lieu tong hop y = X theta_true + eps.
    Thu tu sinh (default_rng(seed)): X_raw ~ N(0,1) (N x D), theta_true ~ N(0,1) (D+1),
    nhieu eps ~ N(0, sigma^2) (N). Tra ve (X co cot 1, y, theta_true)."""
    rng = np.random.default_rng(seed)
    X = add_bias(rng.normal(size=(N, D)))
    theta_true = rng.normal(size=D + 1)
    y = X @ theta_true + sigma * rng.normal(size=N)
    return X, y, theta_true


def heavy_ball_root_modulus(gamma, beta, lam):
    """Mo-dun lon nhat cua nghiem z^2 - (1+beta-gamma*lam) z + beta = 0."""
    b = 1 + beta - gamma * lam
    disc = b * b - 4 * beta
    if disc <= 0:
        return np.sqrt(beta)
    r = np.sqrt(disc)
    return max(abs((b + r) / 2), abs((b - r) / 2))


def heavy_ball_discriminant(gamma, beta, lam):
    return (1 + beta - gamma * lam) ** 2 - 4 * beta


def heavy_ball_gamma_limit(lmax, beta):
    """Dieu kien on dinh heavy-ball: 0 < gamma < 2(1+beta)/lambda_max."""
    return 2 * (1 + beta) / lmax


def poly2_features(X):
    """Dac trung bac 2: x_i, x_i*x_j (i<=j). Khong gom cot 1."""
    n, d = X.shape
    cols = [X] + [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i, d)]
    return np.hstack(cols)
