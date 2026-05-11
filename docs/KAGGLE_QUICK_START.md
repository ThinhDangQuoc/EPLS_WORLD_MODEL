# Kaggle Quick Start - 2-Minute Setup

**TL;DR** - Run this on Kaggle and let it handle timeouts automatically.

## 🚀 In Kaggle Notebook

### Cell 1: Copy Script
```python
# Copy contents of epls_kaggle.py into this cell
```

### Cell 2: Run (Full Pipeline)
```python
!python epls_kaggle.py
```

### Cell 2 (Alternative): Run Phase 1 Only
```python
import os
os.environ["EPLS_RUN_PHASE"] = "1"
!python epls_kaggle.py
```

### Cell 2 (Alternative): Run Phase 2 Only
```python
import os
os.environ["EPLS_RUN_PHASE"] = "2"
!python epls_kaggle.py
```

## ⏱️ Expected Times

| Phase | Tasks | Time | GPU |
|-------|-------|------|-----|
| 1 | Data, VAE, MDRNN, Benchmark | 8-10h | T4/P100 |
| 2 | Setup, Iterate×5, Benchmark | 8-12h | T4/P100 |
| **Total** | - | **16-22h** | - |

## 🔄 If Kaggle Times Out

1. **Just run again** - Script detects completed tasks
2. **It automatically resumes** from where it stopped
3. **No manual work needed**

Example:
```python
# Session 1: Timeout at 9 hours
# Task 1A: ✅ Done
# Task 1B: ✅ Done  
# Task 1C: ✅ Done
# Task 1D: ❌ Only 30% done

# Session 2: Run again
# Task 1A: ⏭️ Skip (already done)
# Task 1B: ⏭️ Skip (already done)
# Task 1C: ⏭️ Skip (already done)
# Task 1D: 🚀 Run fresh (all 100 trials)
```

## 📊 Check Progress

```python
# Add cell to check status:
import json
with open("kaggle_progress.json") as f:
    progress = json.load(f)
print(json.dumps(progress, indent=2))
```

Output:
```json
{
  "phase": 1,
  "stage": "1B_vae_trained",
  "completed_tasks": [
    "phase1_generate_data",
    "phase1_train_vae"
  ],
  "timestamp": "2024-05-11 03:45:00"
}
```

## 📁 Results

After completion, check:
- **Models**: `mdrnn/checkpoints/` & `vae/checkpoints/`
- **Benchmarks**: `tests_custom/planning_test_results/`
- **Progress**: `kaggle_progress.json`

## 🎯 Expected Scores

| Model | Target |
|-------|--------|
| Phase 1 (Random) | ~356 ± 177 |
| Phase 2 (Iterative) | ~708 ± 195 |

## ❌ Common Issues

### Issue: "ModuleNotFoundError: gymnasium"
**Fix**: Automatically installed by script. If fails, restart kernel.

### Issue: "CUDA out of memory"
**Fix**: Reduce batch sizes in script (lines ~220, ~240)

### Issue: Want to restart Phase 1
**Fix**: 
```python
import os
if os.path.exists("kaggle_progress.json"):
    os.remove("kaggle_progress.json")
!python epls_kaggle.py
```

## 📚 Full Docs

For detailed info, see:
- **KAGGLE_NOTEBOOK_GUIDE.md** - Full guide
- **KAGGLE_RESUMPTION_GUIDE.md** - Progress tracking details
- **EXECUTION_GUIDE.md** - Local execution (same code, different environment)

## 🔑 Key Points

✅ **Automatic timeout recovery** - No manual intervention  
✅ **Progress tracking** - Knows exactly what's done  
✅ **Skip completed tasks** - Never runs twice  
✅ **Phases 1 & 2** - Complete workflow  
✅ **GPU-optimized** - Uses Kaggle's T4/P100  

## 🎓 What It Does

1. **Phase 1** (8-10h): Train baseline random model
   - Generate 5,000 rollouts
   - Train VAE
   - Train MDRNN
   - Benchmark with 100 trials

2. **Phase 2** (8-12h): Iterative refinement
   - Copy baseline model
   - Run 5 iterations of improvement
   - Benchmark with 100 trials

## 📞 Need Help?

- **Setup issues**: Check KAGGLE_NOTEBOOK_GUIDE.md
- **Resumption questions**: Check KAGGLE_RESUMPTION_GUIDE.md
- **Code questions**: Check epls_kaggle.py comments
- **Algorithm questions**: Check README.md and paper

---

**That's it!** Run `!python epls_kaggle.py` and come back in ~20 hours. 🚀
