"""Chay lai toan bo thuc nghiem cua BaoCao.pdf (Muc 4-5) va sinh bang/hinh.

    python run_experiments.py

Dau ra (trong src/output/):
    figures/fig1..fig4 + fig5_tuong_quan + fig6_phan_du
    tables/*.csv, results.txt, doi_chieu_bao_cao.csv
Moi ngau nhien deu co hat giong co dinh (xem gd_linreg.SEED, SEED_GRADCHECK).
sklearn CHI duoc import o muc 4.6 de doi chieu; loi thuat toan chi dung numpy.
"""
import os, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gd_linreg import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "winequality-red.csv")
OUT = os.path.join(HERE, "output")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
TOL = 1e-6

_log = io.StringIO()
def log(*a):
    s = " ".join(str(x) for x in a); print(s); _log.write(s + "\n")

def save_csv(name, header, rows):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name)); plt.close()


# ------------------------------------------------------------------ 4.1
log("=" * 70, "\n4.1 DU LIEU, CHIA DU LIEU", "\n" + "=" * 70)
X, y, names = load_wine(DATA)
log(f"X: {X.shape}, y: {y.shape}, thieu gia tri: {int(np.isnan(X).sum() + np.isnan(y).sum())}")
Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, SEED)
log(f"Chia: hoan vi default_rng({SEED}), 320 phan tu dau = kiem tra. Huan luyen {len(ytr)}, kiem tra {len(yte)}")
baseline = mse(yte, ytr.mean())
log(f"Moc so sanh (trung binh nhan): MSE kiem tra = {baseline:.4f}")

m, s = standardize_fit(Xtr)
Ztr, Zte = add_bias(standardize_apply(Xtr, m, s)), add_bias(standardize_apply(Xte, m, s))
Rtr, Rte = add_bias(Xtr), add_bias(Xte)
th_star = normal_equation(Ztr, ytr)
th_star_raw = normal_equation(Rtr, ytr)
lmax, lmin, kappa = hessian_spectrum(Ztr)
lmax_r, lmin_r, kappa_r = hessian_spectrum(Rtr)
Lstar, Lstar_raw = loss(th_star, Ztr, ytr), loss(th_star_raw, Rtr, ytr)
e0 = np.linalg.norm(th_star)
log(f"Chuan hoa: lmax={lmax:.4f} lmin={lmin:.4f} kappa={kappa:.2f}; L*={Lstar:.7f}; ||theta*||={e0:.3f}")

ev_raw = np.sort(np.linalg.eigvalsh(hessian(Rtr)))[::-1]
ev_std = np.sort(np.linalg.eigvalsh(hessian(Ztr)))[::-1]
vals, cnts = np.unique(y, return_counts=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(vals, cnts); ax[0].set_xlabel("Điểm chất lượng (quality)"); ax[0].set_ylabel("Số mẫu"); ax[0].set_title("Phân bố nhãn")
i = np.arange(1, 13)
ax[1].semilogy(i, np.abs(ev_raw), "o-", label="Thô"); ax[1].semilogy(i, ev_std, "s-", label="Chuẩn hoá")
ax[1].set_xlabel("Chỉ số giá trị riêng (giảm dần)"); ax[1].set_ylabel(r"$\lambda_i(\mathbf{H})$")
ax[1].set_title("Phổ của Hessian H"); ax[1].legend(); savefig("fig1_nhan_va_pho.png")

# ------------------------------------------------------------------ 4.2
log("\n" + "=" * 70, "\n4.2 KIEM THU THEO THU TU TANG DAN", "\n" + "=" * 70)
x0 = np.array([0., 1., 2., 3.]); y0 = np.array([1., 3., 4., 8.]); X0 = add_bias(x0[:, None])
th_hand = np.array([0.7, 2.2])
l0max, l0min, k0 = hessian_spectrum(X0)
th0, _ = gradient_descent(X0, y0, 1 / l0max, T=5000)
log(f"(0) Vi du giai tay 4 diem: theta* giai tay = {th_hand}, chuan tac numpy = {normal_equation(X0, y0)}, "
    f"GD = {th0}; |GD - giai tay|_inf = {np.abs(th0 - th_hand).max():.1e}; lambda=({l0min:.4f}, {l0max:.4f})")
gc_err = gradient_check(Ztr, ytr)
log(f"(1) Kiem tra gradient (theta~N(0,I), seed={SEED_GRADCHECK}, eps={EPS_FD}): sai khac max = {gc_err:.2e}")
Xs, ys, th_true = make_synthetic()
th_ne = normal_equation(Xs, ys); lsm = hessian_spectrum(Xs)[0]
th_gd, _ = gradient_descent(Xs, ys, 1 / lsm, T=20000)
syn_gd_ne, syn_true = np.abs(th_gd - th_ne).max(), np.abs(th_ne - th_true).max()
log(f"(2) Tong hop (seed={SEED}, N=500, D=11, X~N(0,1), theta_true~N(0,1), sigma=0.05):")
log(f"    |GD - chuan tac|_inf = {syn_gd_ne:.1e};  |chuan tac - theta_true|_inf = {syn_true:.4f}  (sigma/sqrt(N) = {0.05/np.sqrt(500):.4f})")
log(f"(3) Du lieu that: xem cac muc sau. rank(X) = {np.linalg.matrix_rank(Ztr)} = D+1 = 12; lmin = {lmin:.4f} > 0 (Menh de 1)")

# ------------------------------------------------------------------ 4.3
log("\n" + "=" * 70, "\n4.3 DIEU KIEN SO", "\n" + "=" * 70)
save_csv("bang3_pho_hessian.csv", ["du_lieu", "lmax", "lmin", "kappa", "gamma_max"],
         [["tho", lmax_r, lmin_r, kappa_r, 2 / lmax_r], ["chuan_hoa", lmax, lmin, kappa, 2 / lmax]])
log(f"Tho      : lmax={lmax_r:.3e} lmin={lmin_r:.3e} kappa={kappa_r:.3e} gmax={2/lmax_r:.3e}")
log(f"Chuan hoa: lmax={lmax:.4f} lmin={lmin:.4f} kappa={kappa:.2f} gmax={2/lmax:.4f};  kappa giam {kappa_r/kappa:.2e} lan")
th_std, h_std = gradient_descent(Ztr, ytr, 1 / lmax, T=5000, theta_star=th_star, tol=TOL)
th_raw, h_raw = gradient_descent(Rtr, ytr, 1 / lmax_r, T=5000, theta_star=th_star_raw, tol=TOL)
raw_gap, raw_dist = h_raw["loss"][-1] - Lstar_raw, h_raw["dist"][-1]
raw_mse, raw_opt = mse(yte, Rte @ th_raw), mse(yte, Rte @ th_star_raw)
log(f"Chuan hoa: {h_std['iters_to_tol']} vong den ||theta-theta*||<1e-6")
log(f"Tho (T=5000): L-L*={raw_gap:.3f}, ||theta_T-theta*||={raw_dist:.1f}, MSE kiem tra={raw_mse:.4f} (toi uu {raw_opt:.4f})")
log(f"Can kappa*ln(||e0||/eps) cho du lieu tho ~ {kappa_r*np.log(np.linalg.norm(th_star_raw)/TOL):.2e} vong")
plt.figure(figsize=(6.5, 4))
plt.semilogy(np.maximum(np.array(h_std["loss"]) - Lstar, 1e-16), label=f"Chuẩn hoá (κ={kappa:.1f}, γ=1/λmax)")
plt.semilogy(np.maximum(np.array(h_raw["loss"]) - Lstar_raw, 1e-16), label=f"Dữ liệu thô (κ={kappa_r:.1e}, γ=1/λmax)")
plt.ylim(1e-16, None); plt.xlabel("Số vòng lặp t"); plt.ylabel(r"$L(\theta_t)-L(\theta^*)$"); plt.legend()
savefig("fig2_chuan_hoa_vs_tho.png")

# ------------------------------------------------------------------ 4.4
log("\n" + "=" * 70, "\n4.4 KHAO SAT BUOC HOC", "\n" + "=" * 70)
cs = [0.1, 0.5, 1.0, 1.9, 1.99, 2.05]
runs, rows = {}, []
for c in cs:
    g = c / lmax
    _, h = gradient_descent(Ztr, ytr, g, T=6000, theta_star=th_star, tol=TOL)
    runs[c] = h
    L300 = h["loss"][300] if not h["diverged"] and len(h["loss"]) > 300 else "phan_ky"
    rows.append([f"{c}/lmax", c / 2, g, L300, h["iters_to_tol"] if h["iters_to_tol"] else "—"])
    log(f"c={c:<5} gamma={g:.4f} gamma/gmax={c/2:.3f}  L(theta_300)={L300 if isinstance(L300,str) else f'{L300:.6f}'}  vong={rows[-1][-1]}")
save_csv("bang4_buoc_hoc.csv", ["gamma", "gamma_over_gmax", "gia_tri", "L300", "vong_den_1e-6"], rows)
g_opt = 2 / (lmax + lmin); rho_opt = (kappa - 1) / (kappa + 1)
_, h_opt = gradient_descent(Ztr, ytr, g_opt, T=5000, theta_star=th_star, tol=TOL)
bound_opt = np.log(e0 / TOL) / np.log(1 / rho_opt); bound_inv = kappa * np.log(e0 / TOL)
log(f"gamma_opt={g_opt:.4f} (={g_opt/(2/lmax):.3f} gmax), rho_opt={rho_opt:.4f}; can {bound_opt:.0f} vong, do {h_opt['iters_to_tol']} vong")
log(f"gamma=1/lmax: can kappa*ln ~ {bound_inv:.0f} vong (||e0||={e0:.2f}), do {runs[1.0]['iters_to_tol']} vong")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for c in cs:
    h = runs[c]
    ax[0].semilogy(np.maximum(np.array(h["loss"][:301]) - Lstar, 1e-14), label=fr"$\gamma$={c}/$\lambda_{{max}}$")
    ax[1].semilogy(np.maximum(np.array(h["dist"][:301]), 1e-14), label=fr"$\gamma$={c}/$\lambda_{{max}}$")
ax[0].set_title("Sai số hàm mất mát"); ax[0].set_ylabel(r"$L(\theta_t)-L(\theta^*)$")
ax[1].set_title("Khoảng cách tới nghiệm chuẩn tắc"); ax[1].set_ylabel(r"$\|\theta_t-\theta^*\|$")
for a in ax: a.set_xlabel("Số vòng lặp t"); a.set_ylim(1e-14, 1e7)
ax[0].legend(fontsize=7); savefig("fig3_buoc_hoc.png")

# Tieu chi dung ||g|| < eps  <->  ||theta - theta*|| <= ||g|| / lmin   (vi g = H e)
log("\nTieu chi dung ||g||<eps (Thuat toan 1) so voi ||theta-theta*|| (cac bang):")
for eps in [1e-3, 1e-5, 1e-7]:
    t, _ = gradient_descent(Ztr, ytr, 1 / lmax, T=100000, eps=eps)
    gn, dn = np.linalg.norm(grad(t, Ztr, ytr)), np.linalg.norm(t - th_star)
    log(f"  eps={eps:.0e}: ||g||={gn:.2e}, ||theta-theta*||={dn:.2e} <= ||g||/lmin={gn/lmin:.2e}  [{dn <= gn/lmin + 1e-15}]")

# ------------------------------------------------------------------ 4.5
log("\n" + "=" * 70, "\n4.5 QUAN TINH", "\n" + "=" * 70)
g_hb, b_hb = heavy_ball_params(lmax, lmin)
cfg = [("GD (b=0, g=1/lmax)", 1 / lmax, 0.0), ("Momentum (b=0.5, g=1/lmax)", 1 / lmax, 0.5),
       ("Momentum (b=0.9, g=1/lmax)", 1 / lmax, 0.9), ("Heavy-ball toi uu (Polyak)", g_hb, b_hb)]
mruns, rows = [], []
for nm, g, b in cfg:
    _, h = gradient_descent(Ztr, ytr, g, b, T=400, theta_star=th_star, tol=TOL)
    _, hl = gradient_descent(Ztr, ytr, g, b, T=2000, theta_star=th_star, tol=TOL)
    mruns.append(h); it = h["iters_to_tol"]
    rows.append([nm, g, b, it if it else f">400 ({hl['iters_to_tol']})"]); log(f"{nm}: gamma={g:.4f} beta={b:.3f} vong={rows[-1][-1]}")
save_csv("bang5_quan_tinh.csv", ["cau_hinh", "gamma", "beta", "vong_den_1e-6"], rows)
hb_est = np.log(e0 / TOL) / np.log(1 / np.sqrt(b_hb))
log(f"Heavy-ball toi uu: gamma={g_hb:.4f}, beta={b_hb:.4f}, sqrt(beta)={np.sqrt(b_hb):.4f}; uoc luong tiem can {hb_est:.1f} vong, do {mruns[3]['iters_to_tol']} vong")
lim = heavy_ball_gamma_limit(lmax, b_hb)
log(f"Dieu kien on dinh heavy-ball: gamma={g_hb:.4f} < 2(1+beta)/lmax = {lim:.4f} [{g_hb < lim}]  (trong khi 2/lmax = {2/lmax:.4f}; "
    f"dinh ly 1 chi ap dung cho beta=0)")
lams = np.linalg.eigvalsh(hessian(Ztr))
rm = [heavy_ball_root_modulus(g_hb, b_hb, l) for l in lams]
log(f"Mo-dun nghiem dac trung theo tung tri rieng (12 gia tri): min={min(rm):.6f}, max={max(rm):.6f}; sqrt(beta)={np.sqrt(b_hb):.6f}")
log(f"Biet thuc tai lmin: {heavy_ball_discriminant(g_hb, b_hb, lmin):.2e}, tai lmax: {heavy_ball_discriminant(g_hb, b_hb, lmax):.2e} "
    f"(≈0 -> nghiem kep -> them thua so (1+t); vi vay do 57 > uoc luong 56, day la uoc luong tiem can chu khong phai can tren cung)")
for b in (0.5, 0.9):
    r = [heavy_ball_root_modulus(1 / lmax, b, l) for l in lams]
    log(f"beta={b}, gamma=1/lmax: mo-dun lon nhat = {max(r):.4f} (tri rieng nho nhat), so phuc ung voi {sum(heavy_ball_discriminant(1/lmax,b,l)<0 for l in lams)}/12 tri rieng")
plt.figure(figsize=(7, 4))
for lab, h in zip(["GD (β=0)", "Momentum β=0.5", "Momentum β=0.9", f"Heavy-ball tối ưu (β={b_hb:.3f})"], mruns):
    plt.semilogy(np.maximum(h["dist"], 1e-15), label=lab)
plt.xlabel("Số vòng lặp t"); plt.ylabel(r"$\|\theta_t-\theta^*\|$"); plt.legend(); savefig("fig4_quan_tinh.png")

# ------------------------------------------------------------------ 4.6
log("\n" + "=" * 70, "\n4.6 SO SANH VOI THU VIEN CHUAN", "\n" + "=" * 70)
from sklearn.linear_model import LinearRegression  # CHI DE DOI CHIEU
th_gd2000, _ = gradient_descent(Ztr, ytr, 1 / lmax, T=2000)
th_hb2000, _ = gradient_descent(Ztr, ytr, g_hb, b_hb, T=2000)
lr = LinearRegression().fit(Ztr[:, 1:], ytr); th_sk = np.r_[lr.intercept_, lr.coef_]; pred_sk = lr.predict(Zte[:, 1:])
inf = lambda a, b: float(np.abs(a - b).max())
rows = [["chuan_tac", mse(ytr, Ztr @ th_star), mse(yte, Zte @ th_star), r2(yte, Zte @ th_star), 0.0],
        ["gd_2000", mse(ytr, Ztr @ th_gd2000), mse(yte, Zte @ th_gd2000), r2(yte, Zte @ th_gd2000), inf(th_gd2000, th_star)],
        ["heavy_ball_2000", "", "", "", inf(th_hb2000, th_star)],
        ["sklearn", "", mse(yte, pred_sk), r2(yte, pred_sk), inf(th_sk, th_star)]]
save_csv("bang6_so_sanh.csv", ["phuong_phap", "mse_train", "mse_test", "r2_test", "inf_theta_theta_star"], rows)
for r in rows: log(r)
gd_vs_sk = inf(th_gd2000, th_sk); log(f"|GD - sklearn|_inf = {gd_vs_sk:.1e}")

# ------------------------------------------------------------------ 5
log("\n" + "=" * 70, "\n5. THAO LUAN — BANG CHUNG CHO CAC NHAN DINH", "\n" + "=" * 70)
yhat = Zte @ th_star; rmse = float(np.sqrt(mse(yte, yhat)))
log(f"R2 kiem tra={r2(yte, yhat):.4f}; RMSE={rmse:.3f}; do lech chuan nhan={y.std():.3f}; ty le nhan 5/6 = {np.mean((y==5)|(y==6)):.3f}")
coef = th_star[1:]; order = np.argsort(-np.abs(coef))
log("He so (dac trung chuan hoa):"); [log(f"   {names[k]:<22}{coef[k]:+.3f}") for k in order]
save_csv("he_so.csv", ["dac_trung", "he_so"], [[names[k], round(coef[k], 4)] for k in order])

# (a) Da cong tuyen: tuong quan giua cac dac trung & chi so tuong quan lon nhat
C = np.corrcoef(Xtr, rowvar=False)
iu = np.triu_indices(11, 1); big = sorted(zip(np.abs(C[iu]), iu[0], iu[1]), reverse=True)[:6]
log("\nCac cap dac trung tuong quan manh nhat (|r|):")
[log(f"   {names[a]} ~ {names[b]}: r={C[a,b]:+.3f}") for _, a, b in big]
d = names.index("density")
log(f"   Ti trong: r(alcohol)={C[d,names.index('alcohol')]:+.3f}, r(fixed acidity)={C[d,names.index('fixed acidity')]:+.3f}")
# VIF = phan tu cheo cua C^{-1}
vif = np.diag(np.linalg.inv(C)); log("VIF:", ", ".join(f"{names[k]}={vif[k]:.2f}" for k in np.argsort(-vif)[:4]), "(...)")
save_csv("tuong_quan.csv", ["dac_trung"] + names, [[names[i]] + [round(C[i, j], 4) for j in range(11)] for i in range(11)])
plt.figure(figsize=(6.5, 5.5)); plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar(label="Hệ số tương quan")
plt.xticks(range(11), names, rotation=90, fontsize=7); plt.yticks(range(11), names, fontsize=7); plt.grid(False)
plt.title("Ma trận tương quan (tập huấn luyện)"); savefig("fig5_tuong_quan.png")

# (b) Phi tuyen: them dac trung bac 2 (chuan hoa lai theo thong ke tap huan luyen)
Ptr, Pte = poly2_features(standardize_apply(Xtr, m, s)), poly2_features(standardize_apply(Xte, m, s))
pm, ps = standardize_fit(Ptr)
P1, P2 = add_bias(standardize_apply(Ptr, pm, ps)), add_bias(standardize_apply(Pte, pm, ps))
th_p = normal_equation(P1, ytr)
log(f"\nDac trung bac 2 ({P1.shape[1]-1} dac trung, nghiem chuan tac): MSE huan luyen={mse(ytr,P1@th_p):.4f}, "
    f"MSE kiem tra={mse(yte,P2@th_p):.4f}, R2 kiem tra={r2(yte,P2@th_p):.4f}  (tuyen tinh: {mse(yte,yhat):.4f}, {r2(yte,yhat):.4f})")

# (c) Nhan thu bac & mat can bang: du doan co xu huong co ve trung binh
log("\nTrung binh du doan theo nhan that (tap kiem tra):")
cls = np.unique(yte); mean_pred = [float(yhat[yte == c].mean()) for c in cls]
[log(f"   quality={int(c)} (n={int((yte==c).sum()):>3}): TB du doan = {mp:.3f}") for c, mp in zip(cls, mean_pred)]
res = yte - yhat; log(f"Phan du: TB={res.mean():+.3f}, do lech chuan={res.std():.3f}")
rd = np.clip(np.rint(yhat), 3, 8)
log(f"Lam tron du doan: dung chinh xac {np.mean(rd==yte):.3f}; sai lech <=1 diem {np.mean(np.abs(rd-yte)<=1):.3f}; "
    f"(moc: luon doan 5 -> dung {np.mean(yte==5):.3f}; luon doan 6 -> dung {np.mean(yte==6):.3f})")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].scatter(yte + np.random.default_rng(SEED).uniform(-.12, .12, len(yte)), yhat, s=12, alpha=.6)
ax[0].plot([3, 8], [3, 8], "r--", label="y = ŷ"); ax[0].set_xlabel("Điểm thật"); ax[0].set_ylabel("Điểm dự đoán"); ax[0].legend()
ax[1].hist(res, bins=25); ax[1].set_xlabel("Phần dư y − ŷ"); ax[1].set_ylabel("Số mẫu"); savefig("fig6_phan_du.png")

# ------------------------------------------------------------------ Doi chieu
log("\n" + "=" * 70, "\nDOI CHIEU VOI BAO CAO", "\n" + "=" * 70)
it = lambda h: h["iters_to_tol"]
def chk(name, rep, got, rel=None, ab=None):
    ok = abs(got - rep) <= (ab if ab is not None else rel * abs(rep))
    return [name, rep, got, "OK" if ok else "LECH"]
R = [chk("MSE moc", .6040, baseline, ab=5e-5), chk("lmax chuan hoa", 3.084, lmax, ab=5e-4), chk("lmin chuan hoa", .0597, lmin, ab=5e-5),
     chk("kappa chuan hoa", 51.6, kappa, ab=.05), chk("lmax tho", 3.63e3, lmax_r, rel=.005), chk("lmin tho", 2.82e-7, lmin_r, rel=.01),
     chk("kappa tho", 1.29e10, kappa_r, rel=.01), chk("gamma_max", .6485, 2 / lmax, ab=5e-5), chk("L(theta*)", .2112205, Lstar, ab=5e-7),
     chk("||theta*||", 5.67, e0, ab=.005), chk("vong gamma=1/lmax", 507, it(h_std), ab=0), chk("Tho: L-L*", .030, raw_gap, ab=5e-4),
     chk("Tho: ||theta_T-theta*||", 57.4, raw_dist, ab=.05), chk("Tho: MSE kiem tra", .5001, raw_mse, ab=5e-5),
     chk("MSE kiem tra toi uu", .4018, raw_opt, ab=5e-5),
     chk("B4 L300 c=0.1", .211270, runs[.1]["loss"][300], ab=5e-7), chk("B4 L300 c=0.5", .211221, runs[.5]["loss"][300], ab=5e-7),
     chk("B4 L300 c=1.99", .211233, runs[1.99]["loss"][300], ab=5e-7),
     chk("B4 vong c=0.1", 5114, it(runs[.1]), ab=0), chk("B4 vong c=0.5", 1019, it(runs[.5]), ab=0), chk("B4 vong c=1", 507, it(runs[1.0]), ab=0),
     chk("B4 vong c=1.9", 265, it(runs[1.9]), ab=0), chk("B4 vong c=1.99", 1090, it(runs[1.99]), ab=0),
     chk("gamma_opt", .6362, g_opt, ab=5e-5), chk("rho_opt", .9620, rho_opt, ab=5e-5), chk("gamma_opt can", 401, bound_opt, ab=1),
     chk("gamma_opt do", 285, it(h_opt), ab=0), chk("can kappa*ln", 803, bound_inv, ab=1),
     chk("HB gamma", .9995, g_hb, ab=5e-5), chk("HB beta", .571, b_hb, ab=5e-4), chk("HB sqrt(beta)", .756, np.sqrt(b_hb), ab=5e-4),
     chk("B5 vong b=0.5", 242, it(mruns[1]), ab=0), chk("B5 vong b=0.9", 263, it(mruns[2]), ab=0), chk("B5 vong HB", 57, it(mruns[3]), ab=0),
     chk("HB uoc luong", 56, hb_est, ab=1), chk("B6 MSE train", .4224, mse(ytr, Ztr @ th_gd2000), ab=5e-5),
     chk("B6 R2 test", .3208, r2(yte, Zte @ th_gd2000), ab=5e-5), chk("RMSE", .634, rmse, ab=5e-4), chk("std nhan", .81, y.std(), ab=.005),
     chk("he so con", .289, coef[names.index("alcohol")], ab=5e-4), chk("he so axit bay hoi", -.198, coef[names.index("volatile acidity")], ab=5e-4),
     chk("he so sunfat", .148, coef[names.index("sulphates")], ab=5e-4), chk("he so axit co dinh", .115, coef[names.index("fixed acidity")], ab=5e-4),
     chk("he so SO2 tong", -.102, coef[names.index("total sulfur dioxide")], ab=5e-4)]
save_csv("doi_chieu_bao_cao.csv", ["muc", "bao_cao", "tinh_duoc", "ket_qua"], R)
for r in R: log(f"  [{r[3]:<4}] {r[0]:<26} bao cao={r[1]:<10} tinh duoc={r[2]:.6g}")
log(f"\nTong: {sum(r[3]=='OK' for r in R)}/{len(R)} muc khop.")
log(f"Sai so may: GD={inf(th_gd2000,th_star):.1e} HB={inf(th_hb2000,th_star):.1e} sklearn={inf(th_sk,th_star):.1e} GD-sklearn={gd_vs_sk:.1e}")
log("\nCac so KHONG co trong bao cao goc (do bao cao khong neu cach sinh) - cap nhat bao cao theo:")
log(f"   kiem tra gradient = {gc_err:.1e} (bao cao: 3.6e-10); tong hop GD-NE = {syn_gd_ne:.1e} (bao cao: 6.7e-16); tong hop NE-theta_true = {syn_true:.4f} (bao cao: 0.0078)")
open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8").write(_log.getvalue())
print(f"\nDa ghi: {OUT}")
