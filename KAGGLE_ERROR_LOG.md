# Kaggle/Python 3.12 Error Log - EPLS Reproduction

File này ghi lại các lỗi phát sinh trong quá trình chạy project EPLS trên môi trường Kaggle (Python 3.12, NumPy 2.0+) và các giải pháp đã thực hiện.

## 1. Lỗi Tương Thích Gym 0.21.0 (Metadata Generation Failed)
- **Triệu chứng:** `pip install gym==0.21.0` thất bại trên Python 3.12.
- **Nguyên nhân:** Bản Gym cũ không hỗ trợ build trên Python mới.
- **Giải pháp:** Chuyển sang dùng `Gymnasium` (v2) và sử dụng `StepCompatibilityWrapper` trong `epls_kaggle.py` để ép giá trị trả về của `step()` từ 5 về 4 giá trị.

## 2. Lỗi NumPy 2.0 (AttributeError: np.float)
- **Triệu chứng:** `AttributeError: module 'numpy' has no attribute 'float'`.
- **Nguyên nhân:** NumPy 2.0 đã khai tử các alias như `np.float`, `np.int`, `np.bool`.
- **Giải pháp:** Chạy script tự động thay thế toàn bộ project: `np.float` -> `float`, `np.int` -> `int`.

## 3. Lỗi Tràn Bộ Nhớ Đĩa (Kaggle Disk Space Limit 20GB)
- **Triệu chứng:** Quá trình gen data bị dừng đột ngột hoặc Notebook bị treo.
- **Nguyên nhân:** 10,000 rollouts chiếm quá nhiều dung lượng (~10-15GB).
- **Giải pháp:** Giảm số lượng rollout xuống còn 5,000 cho mỗi giai đoạn (Random/Expert).

## 4. Lỗi Thiếu Box2D (DependencyNotInstalled)
- **Triệu chứng:** `Box2D is not installed, you can install it by run pip install swig...`.
- **Nguyên nhân:** Môi trường Kaggle thiếu engine vật lý cho CarRacing.
- **Giải pháp:** Cài đặt `swig` qua `apt-get` và `pip`, sau đó ép cài `gymnasium[box2d]`.

## 5. Lỗi Đồ Họa (ImportError: Qt5 bindings)
- **Triệu chứng:** `Failed to import any of the following Qt binding modules: PyQt5, PySide2`.
- **Nguyên nhân:** Code gọi `matplotlib.use('Qt5Agg')` trong môi trường server không có màn hình (headless).
- **Giải pháp:** Đổi backend matplotlib sang `Agg` trong file `simulated_environment.py`.

## 6. Lỗi Package Python (ModuleNotFoundError: No module named 'mdrnn.iteration_stats')
- **Triệu chứng:** Không thể import các module bên trong project dù file đã tồn tại.
- **Nguyên nhân:** Thiếu file `__init__.py` và lỗi cache module của Python (`sys.modules`).
- **Giải pháp:** Thêm hàm `fix_python_packages()` vào notebook để tự động tạo `__init__.py` và xóa cache module.

## 7. Lỗi Xóa Symlink (OSError: Cannot call rmtree on a symbolic link)
- **Triệu chứng:** Lỗi khi chạy lại Notebook lần 2 tại bước setup assets.
- **Nguyên nhân:** Python 3.12 không cho phép `shutil.rmtree` xóa các liên kết mềm (symlink).
- **Giải pháp:** Sử dụng `os.path.islink` và `os.unlink` để xóa link trước khi tạo lại.

## 8. Lỗi Thiếu File Trên Git (.gitignore Over-filtering)
- **Triệu chứng:** `ModuleNotFoundError` dù file đã có ở máy local.
- **Nguyên nhân:** File `iteration_result.py` nằm trong folder bị `.gitignore` chặn.
- **Giải pháp:** Sử dụng `git add -f` để ép Git nhận file code quan trọng.

## 9. Lỗi Gymnasium Logger (AttributeError: set_level)
- **Triệu chứng:** `module 'gymnasium.logger' has no attribute 'set_level'`.
- **Nguyên nhân:** API của Gymnasium khác với Gym cũ.
- **Giải pháp:** Bọc lệnh gọi vào `hasattr(gym.logger, 'set_level')`.

## 10. Lỗi Môi Trường Cũ (DeprecatedEnv: CarRacing-v0)
- **Triệu chứng:** `Environment version v0 for CarRacing is deprecated. Please use CarRacing-v2 instead.`
- **Nguyên nhân:** Gymnasium không còn hỗ trợ tên `v0` mặc định.
- **Giải pháp:** Thực hiện tìm và thay thế hàng loạt (Mass-replace) toàn bộ chuỗi `CarRacing-v0` thành `CarRacing-v2` trong tất cả các file `.py` và `.json`.

## 11. Lỗi Xung Đột Git Pull (config.json conflict)
- **Triệu chứng:** `error: Your local changes to the following files would be overwritten by merge: config.json`.
- **Nguyên nhân:** File `config.json` bị code tự động thay đổi trên Kaggle, gây xung đột khi chạy lệnh `git pull`.
- **Giải pháp:** Thêm lệnh `git checkout -- .` trước khi `git pull` để xóa bỏ các thay đổi tạm thời trên Kaggle.

## 12. Lỗi NumPy 2.0 Trong Tiến Trình Con (Sub-process AttributeError)
- **Triệu chứng:** `AttributeError: module 'numpy' has no attribute 'bool8'` xảy ra ngay cả khi đã vá ở main process.
- **Nguyên nhân:** Khi dùng `multiprocessing` với method `spawn`, các tiến trình con không nhận được bản vá từ tiến trình chính.
- **Giải pháp:** Chèn trực tiếp bản vá NumPy vào các file cốt lõi được Worker import: `main.py` và `base_rollout_generator.py`.

## 13. Lỗi Cú Pháp Docstring (SyntaxError: unterminated triple-quoted string)
- **Triệu chứng:** `SyntaxError: unterminated triple-quoted string literal`.
- **Nguyên nhân:** Sơ suất khi chèn bản vá vào đầu file làm hỏng cấu trúc dấu ngoặc kép ba `"""`.
- **Giải pháp:** Dọn dẹp lại đầu file `main.py`, đảm bảo các dấu `"""` được đóng mở chính xác.

## 14. Lỗi Thiếu Bộ Hiển Thị (AttributeError: 'CarRacing' object has no attribute 'viewer')
- **Triệu chứng:** Crash khi đang gen data: `AttributeError: 'CarRacing' object has no attribute 'viewer'`.
- **Nguyên nhân:** Code gốc cố gắng gọi `viewer.window.dispatch_events()` nhưng trên Kaggle không có màn hình nên `viewer` không tồn tại.
- **Giải pháp:** Bọc lệnh gọi vào khối `try-except` và kiểm tra `hasattr(environment.environment, 'viewer')`.
