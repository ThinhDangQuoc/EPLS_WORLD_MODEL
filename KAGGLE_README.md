# Hướng Dẫn Chạy EPLS trên Kaggle

## Bước 1: Chuẩn bị & Upload Dataset

Nén toàn bộ thư mục `WorldModelPlanning` thành file zip và upload lên Kaggle Dataset:

```bash
# Chạy lệnh này từ thư mục Project/
cd /home/thinh/KHMT/MangNeural/Project
zip -r epls_world_model.zip WorldModelPlanning/ \
    --exclude "WorldModelPlanning/data_random_raw/*" \
    --exclude "WorldModelPlanning/data_iterative/*" \
    --exclude "WorldModelPlanning/.git/*"
```

Sau đó vào https://www.kaggle.com/datasets → **New Dataset** → upload `epls_world_model.zip`
Đặt tên dataset: `epls-world-model`

---

## Bước 2: Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code → **New Notebook**
2. **Settings** → Accelerator: chọn **GPU T4 x2**
3. **Settings** → Internet: **On** (để cài packages)
4. **Add Data** → tìm dataset `epls-world-model` vừa upload

---

## Bước 3: Chạy Notebook

Upload file `epls_kaggle.py` vào notebook (hoặc paste nội dung).

Chạy tuần tự từng cell theo thứ tự:

| Cell | Nội dung | Thời gian T4 |
|------|----------|-------------|
| Cell 1 | Install dependencies | ~2 phút |
| Cell 2 | Verify GPU/env | < 1 phút |
| Cell 3 | Setup project path | < 1 phút |
| Cell 4 | Helper functions + Xvfb | < 1 phút |
| **PHASE 1** | Generate 10k rollouts | ~1–2 giờ |
| **PHASE 2** | Train VAE 50 epochs | ~20–40 phút |
| **PHASE 3** | Train MDRNN 60 epochs | ~1–2 giờ |
| **PHASE 4** | Benchmark baseline (~356) | ~20–30 phút |
| **PHASE 5** | Iterative training × 5 | ~5–8 giờ |
| **PHASE 6** | Final benchmark 100 trials | ~1 giờ |
| Save | Export kết quả | < 1 phút |

**Tổng: ~8–14 giờ** (nằm trong giới hạn 12h/session Kaggle)

> ⚠️ **Nếu session timeout:** Kaggle lưu `/kaggle/working` → chạy lại từ phase bị dừng.
> Code đã có `is_continue_model: true` nên sẽ resume từ checkpoint.

---

## So Sánh Tham Số: Máy Local vs Kaggle

| Tham số | Máy local (CPU) | Kaggle (T4 GPU) |
|---------|----------------|-----------------|
| `mdrnn.hidden_units` | 256 | **512** (đúng paper) |
| `vae_trainer.max_epochs` | 20 | **50** (đúng paper) |
| `mdrnn_trainer.max_epochs` | 30 | **60** (đúng paper) |
| `mdrnn_trainer.batch_size` | 10 | **64** |
| `vae_trainer.batch_size` | 64 | **100** |
| `early_stop_after_n_bad_epochs` | 3 | **5** |
| `fixed_cpu_cores` | 8 | **4** (Kaggle giới hạn) |
| Thời gian ước tính | ~37–62 giờ | **~8–14 giờ** |

---

## Lưu Ý Quan Trọng

1. **Dataset path:** Đổi `DATASET_PATH` trong Cell 3 nếu tên dataset khác
   ```python
   DATASET_PATH = "/kaggle/input/YOUR-DATASET-NAME/WorldModelPlanning"
   ```

2. **Session limit:** Kaggle giới hạn ~12h GPU. Chia thành 2 sessions nếu cần:
   - **Session 1:** Phase 1–4 (generate + train baseline)
   - **Session 2:** Phase 5–6 (iterative + final benchmark)

3. **Lưu kết quả:** Cell "Save Results" copy model + logs vào `/kaggle/working/epls_results/`
   Kaggle tự lưu `/kaggle/working` sau session.

4. **TensorBoard:** Kaggle chưa hỗ trợ native TensorBoard, nhưng logs được lưu để xem sau.
