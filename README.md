# Nhận diện chữ số viết tay bằng CNN trên FPGA (PYNQ-Z2)

Mạng CNN nhỏ (796 tham số) huấn luyện bằng PyTorch trên MNIST, lượng tử hóa int8, cài đặt thành bộ tăng tốc RTL
(Verilog) giao tiếp AXI-Stream + AXI DMA và triển khai trên board **PYNQ-Z2** (Zynq-7000 `xc7z020clg400-1`).

```
ảnh 28x28 (uint8) ─► conv1 5x5x3 ─► maxpool 2x2 + ReLU ─► conv2 5x5 (3→3) ─► maxpool 2x2 + ReLU ─► FC 48→10 ─► argmax
                        24x24x3           12x12x3                8x8x3              4x4x3           10
```

## Kết quả

| Mô hình | Độ chính xác (10 000 ảnh test MNIST) |
|---|---|
| PyTorch float (`model/cnn_mnist.pt`) | 96,35 % |
| **Phần cứng RTL (int8, bộ `.mem` trong dự án Vivado)** | **86,50 %** |

- 1000 ảnh của testbench (`sim/testvector/input_1000.txt`) lấy từ tập **train** MNIST; RTL đạt 90,0 % trên bộ này
  (`sim/results/simulate_1000_axis_cnn_mnist_1000_tb.log`). Đây không phải số liệu tổng quát — dùng 86,5 % ở trên.
- Khoảng chênh với bản float do lượng tử hóa sau huấn luyện (trọng số `trunc(w×128)` int8, không QAT).

Tài nguyên & thời gian (PYNQ-Z2, Vivado 2025.1, xung PL **50 MHz**, `impl_1`):
LUT 21 297 (40,0 %), FF 14 278 (13,4 %), BRAM 2,5 (1,8 %), **DSP 205/220 (93,2 %)**, WNS +4,27 ns, công suất ước tính 1,58 W.
Lõi cần khoảng 1281 chu kỳ / ảnh (~25,6 µs ở 50 MHz).

### Trạng thái kiểm chứng (đọc kỹ)

| Hạng mục | Trạng thái |
|---|---|
| RTL ↔ mô hình số nguyên `tools/golden_model.py` | ✅ khớp 1000/1000 quyết định với log mô phỏng |
| Bitstream PYNQ-Z2 dùng đúng bộ `.mem` trong `vivado/` | ⚠️ suy ra từ log tổng hợp Vivado, chưa đối chiếu trực tiếp |
| Chạy thực tế trên board PYNQ-Z2 | ❌ **chưa thực hiện trong bản này** (không có board). `deploy/pynq_z2/` chưa được chạy thử |
| Testbench `top_tb.v`, `top_tb_1000.v`, `axis_cnn_mnist_tb.v` (đã sửa đường dẫn) | ⚠️ chưa chạy lại; chỉ `axis_cnn_mnist_1000_tb.v` có log mô phỏng |

### Lưu ý về trọng số

Notebook huấn luyện (`model/cnn_mnist.ipynb`) xuất ra bộ `.mem` trong `model/mem_export/`, **khác** bộ `.mem` đang nằm trong
RTL/bitstream. Nếu đưa bộ xuất từ notebook vào RTL thì độ chính xác phần cứng chỉ còn **70,6 %** (đo bằng
`golden_model.py`). Vì vậy repo giữ nguyên RTL + bitstream đã build (86,5 %); bộ `.pt` sinh ra đúng bộ `.mem` trong RTL
không còn trong repo. Việc huấn luyện lại / lượng tử hóa tốt hơn (QAT) rồi build lại bitstream là hướng cải tiến.

## Cấu trúc thư mục

```
model/                  huấn luyện PyTorch, xuất trọng số
  cnn_mnist.ipynb         huấn luyện, lưu .pt, xuất .mem (nhân 128, bù 2, hex)
  base64_test.ipynb       thử suy luận từ ảnh base64 (giống dữ liệu web gửi lên)
  cnn_mnist.pt            mô hình float
  mem_export/             .mem do notebook xuất (KHÁC bộ trong RTL, xem trên)
  bmp/                    20 ảnh MNIST mẫu (train_0..19)
vivado/NhanDienChuVietTay/   dự án Vivado 2025.1 cho PYNQ-Z2 (chỉ mã nguồn)
  *.srcs/sources_1/imports/module   RTL + .mem (bộ trọng số của bitstream)
  *.srcs/sources_1/bd               block design (PS7, AXI DMA, FIFO, axis_cnn_mnist)
  *.srcs/sim_1/imports/testbench    testbench
sim/
  testvector/             input_1000.txt (1000 ảnh), 3_0.txt (1 ảnh chữ số 3)
  results/                log mô phỏng RTL
deploy/pynq_z2/         cnn_mnist.bit + cnn_mnist.hwh, hw_test.ipynb, app/ (Flask demo)
tools/golden_model.py   mô hình số nguyên bit-accurate của RTL (chỉ cần numpy)
```

## Cách dùng

**Kiểm tra độ chính xác phần cứng không cần board/simulator**
```bash
# cần tập MNIST raw: chạy ô tải dữ liệu của model/cnn_mnist.ipynb (tạo model/data/MNIST/raw)
python tools/golden_model.py \
    --mem-dir vivado/NhanDienChuVietTay/NhanDienChuVietTay.srcs/sources_1/imports/module \
    --mnist-dir model/data/MNIST/raw \
    --vectors sim/testvector/input_1000.txt \
    --sim-log sim/results/simulate_1000_axis_cnn_mnist_1000_tb.log
```
Kết quả mong đợi: `86.50%`, `90.0%`, `1000/1000`.

**Windows:** một số file IP trong `vivado/` có đường dẫn dài (~160 ký tự); nếu `git clone` báo *Filename too long*,
chạy `git config --global core.longpaths true` hoặc clone vào thư mục ngắn (ví dụ `C:\cnn`).

**Huấn luyện & xuất trọng số:** `pip install -r requirements.txt`, mở `model/cnn_mnist.ipynb`.

**Mô phỏng RTL (Vivado xsim):** mở `vivado/NhanDienChuVietTay/NhanDienChuVietTay.xpr`
(lần đầu: *Generate Output Products* cho block design). Testbench đọc file dữ liệu theo tên tương đối — chép
`sim/testvector/input_1000.txt` (và `3_0.txt`) vào thư mục chạy mô phỏng
`NhanDienChuVietTay.sim/sim_1/behav/xsim/` trước khi chạy `axis_cnn_mnist_1000_tb`.

**Chạy trên PYNQ-Z2:** chép repo lên board, mở `deploy/pynq_z2/hw_test.ipynb` (nạp bitstream, thử 1 ảnh, đối chiếu 1000 ảnh
với mô phỏng, đo thời gian). Demo web: `python3 deploy/pynq_z2/app/app.py`, mở `http://<ip-board>:5000` và vẽ chữ số.

## Giao diện lõi (`axis_cnn_mnist`)

- Vào: AXI-Stream 8 bit, 784 pixel (0..255) theo hàng, `tlast` không dùng.
- Ra: 1 beat 8 bit (chữ số 0–9) kèm `tlast`.
- Lõi không dùng `m_axis_tready` (không có backpressure); block design có `axis_data_fifo` ở đầu ra để đệm kết quả.

## Nguồn gốc và phần nhóm thực hiện

- **Nguồn:** lõi CNN bằng RTL và quy trình huấn luyện → xuất `.mem` → mô phỏng dựa trên thiết kế của Bo Young Kang
  (weenslab), phát triển từ repo `boaaaang/CNN-Implementation-in-Verilog` (2021). Các header bản quyền trong file `.v`
  được giữ nguyên.
- **Huấn luyện CNN:** dùng nguyên bản gốc (`model/cnn_mnist.ipynb`), không đổi kiến trúc hay thuật toán.
- **Thay đổi của nhóm:** chuyển sang board **PYNQ-Z2** (Zynq-7000, `xc7z020`); bản gốc nhắm tới Kria KV260 (Zynq UltraScale+).
  Vì vậy chỉ phần tích hợp phải làm lại: block design với `processing_system7` thay cho `zynq_ultra_ps_e`, AXI DMA,
  interconnect/SmartConnect, FIFO AXI-Stream, xung PL 50 MHz, build bằng Vivado 2025.1. Lõi CNN (`axis_cnn_mnist`) giữ nguyên.
- **Bổ sung khi chuẩn bị bản phát hành:** `tools/golden_model.py`, notebook và app cho PYNQ-Z2 (`deploy/pynq_z2/`), sửa đường
  dẫn testbench, tài liệu này.

<!-- TODO: chọn và thêm file LICENSE. -->