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

## 15. Lỗi Sai Lệch Tham Số Lấy Mẫu (TypeError: ActionSampler.sample())
- **Triệu chứng:** `TypeError: CarRacingActionSampler.sample() takes 1 positional argument but 2 were given`.
- **Nguyên nhân:** Rollout Generator truyền vào `previous_action` nhưng hàm `sample()` của Sampler không được khai báo nhận tham số này.
- **Giải pháp:** Cập nhật định nghĩa hàm thành `sample(self, previous_action=None)` để tương thích mọi lời gọi.

## 16. Lỗi Tràn Màn Hình Log (Notebook Scrolling Spam)
- **Triệu chứng:** Hàng nghìn dòng thanh tiến trình `tqdm` và thông báo in ra liên tục làm treo trình duyệt.
- **Nguyên nhân:** Sử dụng `tqdm` với `position` trong đa tiến trình trên Notebook không được hỗ trợ tốt; và in thông báo lưu file quá thường xuyên.
- **Giải pháp:** Gỡ bỏ `tqdm` ở vòng lặp trong và ẩn các lệnh `print` kết thúc mỗi Rollout. Chỉ giữ lại log ở mức độ thread/stage.

## 17. Lỗi Hardcode GPU (AcceleratorError: CUDA error: invalid device ordinal)
- **Triệu chứng:** Crash khi bắt đầu huấn luyện MDRNN với lỗi `invalid device ordinal`.
- **Nguyên nhân:** Code trong `mdrnn/mdrnn_trainer.py` hardcode gọi `torch.cuda.set_device(2)`, nhưng máy Kaggle không có đủ 3 GPU.
- **Giải pháp:** Xóa bỏ lệnh `torch.cuda.set_device(2)`, để PyTorch tự động nhận diện và sử dụng thiết bị mặc định (thường là `cuda:0`).

## 18. Lỗi Môi trường bị Khai tử (gymnasium.error.DeprecatedEnv: CarRacing-v2)
- **Triệu chứng:** `gymnasium.error.DeprecatedEnv: Environment version v2 for CarRacing is deprecated. Please use CarRacing-v3 instead.`
- **Nguyên nhân:** Phiên bản Gymnasium mới trên Kaggle đã khai tử `v2` và yêu cầu dùng `v3`.
- **Giải pháp:** Cập nhật `epls_kaggle.py` để tự động kiểm tra xem `CarRacing-v3` có tồn tại không bằng `gym.spec()`. Nếu có thì dùng `v3`, nếu không thì fallback về `v2`. Đồng thời cập nhật `compatible_make` để ánh xạ mọi yêu cầu "CarRacing" về phiên bản khả dụng nhất.

## 19. Lỗi thiếu hàm seed() và thay đổi kết quả reset() (AttributeError: 'StepCompatibilityWrapper' object has no attribute 'seed')
- **Triệu chứng:** `AttributeError: 'StepCompatibilityWrapper' object has no attribute 'seed'` và lỗi unpack khi gọi `reset()`.
- **Nguyên nhân:** Gymnasium (v21+) đã xóa hàm `.seed()` (gộp vào `reset(seed=...)`) và thay đổi kết quả trả về của `.reset()` thành một tuple `(obs, info)`.
- **Giải pháp:** Cập nhật `StepCompatibilityWrapper` trong `epls_kaggle.py`:
    - Thêm hàm `.seed(s)` để lưu lại seed.
    - Ghi đè hàm `.reset()` để truyền seed vào và chỉ trả về `obs` (để tương thích với code cũ).

## 20. Lỗi không thể truy cập thuộc tính môi trường qua Wrapper (AttributeError: 'TimeLimit' object has no attribute 'track')
- **Triệu chứng:** `AttributeError: 'TimeLimit' object has no attribute 'track'` (hoặc `world`, `car`).
- **Nguyên nhân:** Kể từ Gymnasium 0.26, lớp `Wrapper` không còn tự động ủy quyền (delegate) việc truy cập thuộc tính qua hàm `__getattr__`. Do đó, khi môi trường bị bọc bởi `TimeLimit` hoặc `StepCompatibilityWrapper`, các biến đặc thù của `CarRacing` sẽ không thể truy cập trực tiếp.
- **Giải pháp:** Thực hiện monkey-patch cho `gym.Wrapper` trong `epls_kaggle.py` để khôi phục lại hàm `__getattr__` truyền thống, giúp các wrapper tự động tìm kiếm thuộc tính ở môi trường bên trong (inner env).
