# CP Nhóm 1 — Hồi quy tuyến tính bằng hạ gradient (Wine Quality, rượu vang đỏ)

Môn MAI391 — Toán cho Học máy. Lõi thuật toán chỉ dùng **numpy**; `scikit-learn` chỉ dùng ở mục 4.6 để đối chiếu.

## Cách chạy
```bash
cd src
pip install -r requirements.txt
python test_core.py          # 7 kiểm thử: ví dụ giải tay -> gradient -> tổng hợp -> dữ liệu thật
python run_experiments.py    # sinh toàn bộ bảng/hình của báo cáo vào src/output/ (~vài giây)
jupyter notebook CP_Nhom1_GD_LinReg.ipynb    # Run All: bản có giải thích từng bước
```
Dữ liệu: `../data/winequality-red.csv` (UCI, DOI 10.24432/C56S3T, CC BY 4.0), xem `../data/nguon_du_lieu.txt`.

## Các tệp
| Tệp | Nội dung |
|---|---|
| `gd_linreg.py` | Lõi (numpy): hàm mất mát, gradient, Hessian, nghiệm chuẩn tắc, hạ gradient có quán tính (Thuật toán 1), tham số heavy-ball, tiện ích kiểm thử |
| `test_core.py` | Kiểm thử tăng dần (Bước 4 của đề), có ví dụ 4 điểm giải tay |
| `run_experiments.py` | Chạy toàn bộ thực nghiệm Mục 4–5, ghi `output/` và **đối chiếu tự động 44 số liệu** với `BaoCao.pdf` |
| `CP_Nhom1_GD_LinReg.ipynb` | Cùng nội dung ở dạng notebook (đã chạy sẵn, có hình) |
| `output/figures/` | Hình 1–6 (PNG) |
| `output/tables/` | Bảng 3–6, hệ số, ma trận tương quan, bảng đối chiếu (CSV) |
| `output/results.txt` | Toàn bộ log số liệu |

## Tái lập (mọi nguồn ngẫu nhiên)
| Thành phần | Cách cố định |
|---|---|
| Chia dữ liệu | `np.random.default_rng(42).permutation(1599)`; **320 phần tử đầu = tập kiểm tra**, 1279 phần tử còn lại = huấn luyện |
| Chuẩn hoá | trung bình, độ lệch chuẩn tính trên tập huấn luyện, áp cho tập kiểm tra |
| Khởi tạo | θ₀ = 0, v₀ = 0 (không ngẫu nhiên) |
| Kiểm tra gradient | θ ~ N(0, I) từ `default_rng(0)`; sai phân hữu hạn trung tâm, ε = 1e-5 |
| Dữ liệu tổng hợp | `default_rng(42)`: X ~ N(0,1) cỡ 500×11, θ_true ~ N(0,1) (12 phần tử), nhiễu ~ N(0, 0.05²); thứ tự sinh: X, θ_true, nhiễu |

## Ghi chú về số liệu
Báo cáo gốc không nêu cách sinh dữ liệu tổng hợp và θ của kiểm tra gradient, nên ba số dưới đây là kết quả của cấu hình **ghi ở bảng trên**
(cùng bậc với báo cáo; cập nhật số trong báo cáo cho khớp):

| Số liệu | Báo cáo gốc | Code này |
|---|---|---|
| Kiểm tra gradient (sai khác lớn nhất) | 3,6×10⁻¹⁰ | 3,2×10⁻¹⁰ |
| Tổng hợp: ‖GD − chuẩn tắc‖∞ | 6,7×10⁻¹⁶ | 8,9×10⁻¹⁶ |
| Tổng hợp: ‖chuẩn tắc − θ_true‖∞ | 0,0078 | 0,0037 (σ/√N = 0,0022) |

Tất cả 44 số liệu còn lại (Bảng 3–6, hệ số, số vòng lặp, ...) khớp với báo cáo (xem `output/tables/doi_chieu_bao_cao.csv`).
