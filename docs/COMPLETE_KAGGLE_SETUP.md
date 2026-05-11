# Complete Kaggle Setup Guide

Everything you need to run the EPLS project on Kaggle with automatic progress tracking.

## 📦 What You Get

### 2 Executable Files
1. **`epls_kaggle.py`** (510 lines)
   - Python script version
   - Run in terminal: `python epls_kaggle.py`
   - Works on local machine or Kaggle

2. **`epls_kaggle.ipynb`** (27 cells)
   - Jupyter notebook version
   - Run in Kaggle or Jupyter
   - Interactive cell-by-cell execution

### 7 Documentation Files
1. **KAGGLE_INDEX.md** - Master index
2. **KAGGLE_QUICK_START.md** - 2-minute setup
3. **KAGGLE_NOTEBOOK_GUIDE.md** - Complete guide
4. **KAGGLE_RESUMPTION_GUIDE.md** - Progress tracking details
5. **KAGGLE_UPDATES_SUMMARY.md** - What changed from old code
6. **NOTEBOOK_USER_GUIDE.md** - Notebook-specific guide
7. **SCRIPT_VS_NOTEBOOK.md** - Script vs notebook comparison

## 🚀 Quick Start (Choose One)

### Option A: Kaggle Notebook (Recommended)
```
1. Open Kaggle.com → Create New Notebook
2. Click "File" → "Upload notebook"
3. Select: epls_kaggle.ipynb
4. Click "Run All" button
5. Wait 20 hours (across multiple sessions)
6. Check results in test_custom/planning_test_results/
```

### Option B: Kaggle Python Script
```
1. Open Kaggle → Create New Notebook
2. In first cell, paste entire epls_kaggle.py code
3. Run cell
4. Monitoring output in real-time
5. Script automatically handles timeouts
```

### Option C: Local Machine
```bash
# Copy project to local
git clone https://github.com/ThinhDangQuoc/EPLS_WORLD_MODEL.git
cd EPLS_WORLD_MODEL

# Run script
python epls_kaggle.py

# Or use notebook
jupyter notebook epls_kaggle.ipynb
```

## 🎯 Key Features

✅ **Automatic Progress Tracking** - knows what's done  
✅ **Timeout Recovery** - resumes automatically after Kaggle timeouts  
✅ **No Duplicate Work** - each task runs exactly once  
✅ **2 Execution Formats** - script or notebook, your choice  
✅ **7 Documentation Files** - for every situation  
✅ **Gymnasium 1.1.1+** - modern environment library  
✅ **PyTorch 2.x** - latest deep learning framework  
✅ **Phase Control** - run Phase 1 or 2 independently  

## 📊 Expected Results

### Phase 1: Random Baseline (8-10 hours)
```
Data generation:    2-3 hours
VAE training:       1-2 hours
MDRNN training:     2-3 hours
Benchmarking:       2-3 hours
─────────────────────────────
Total: 7-11 hours
Expected score: ~356 ± 177
```

### Phase 2: Iterative Refinement (8-12 hours)
```
Checkpoint setup:   <5 minutes
5 iterations:       6-10 hours
Benchmarking:       2-3 hours
─────────────────────────────
Total: 8-13 hours
Expected score: ~708 ± 195
```

### Total Project: 16-24 hours
(Across multiple Kaggle sessions due to 9-hour timeout)

## 📁 Files Included

```
WorldModelPlanning/
├── epls_kaggle.py                    ← Python script
├── epls_kaggle.ipynb                 ← Jupyter notebook
├── KAGGLE_INDEX.md                   ← Master index
├── KAGGLE_QUICK_START.md             ← Quick start
├── KAGGLE_NOTEBOOK_GUIDE.md          ← Notebook guide
├── KAGGLE_RESUMPTION_GUIDE.md        ← Progress tracking
├── KAGGLE_UPDATES_SUMMARY.md         ← What's new
├── NOTEBOOK_USER_GUIDE.md            ← Notebook details
├── SCRIPT_VS_NOTEBOOK.md             ← Comparison
├── EXECUTION_GUIDE.md                ← Local execution
├── main.py
├── config.json
└── [other project files]
```

## ⚙️ How It Works

### Progress Tracking System
```
kaggle_progress.json (auto-created):
{
  "phase": 1,
  "stage": "1B_vae_trained",
  "completed_tasks": ["phase1_generate_data", "phase1_train_vae"],
  "timestamp": "2024-05-11 03:45:00"
}
```

### Timeout Recovery
```
Session 1: Run tasks 1A, 1B, 1C → TIMEOUT after 9 hours
           Tasks 1A, 1B saved to progress file
           
Session 2: Run again → Skip 1A, 1B → Continue with 1C, 1D
           Script detects completed tasks automatically
           
Session 3: Continue Phase 2 if needed
```

## 🛠️ Customization

### Reduce Training Time
Edit in script or notebook:

```python
# Reduce data
"data_generator": {"rollouts": 1000}  # from 5000

# Reduce trials
"test_suite": {"trials": 50}  # from 100

# Reduce iterations
"iterative_trainer": {"num_iterations": 3}  # from 5
```

### Use Different Planning Agent
```python
"planning_agent": "RHEA"  # Options: RMHC, RHEA, MCTS
```

### Increase Epochs for Better Results
```python
"mdrnn_trainer": {"max_epochs": 100}  # from 60
"vae_trainer": {"max_epochs": 50}  # from 20
```

## 📚 Which Document to Read?

### I want to run it NOW
→ Read: **KAGGLE_QUICK_START.md** (2 minutes)

### I'm using Kaggle Notebook
→ Read: **NOTEBOOK_USER_GUIDE.md** (10 minutes)

### I want complete instructions
→ Read: **KAGGLE_NOTEBOOK_GUIDE.md** (15 minutes)

### I need to understand timeouts
→ Read: **KAGGLE_RESUMPTION_GUIDE.md** (20 minutes)

### I want to compare script vs notebook
→ Read: **SCRIPT_VS_NOTEBOOK.md** (5 minutes)

### I'm running locally
→ Read: **EXECUTION_GUIDE.md** (20 minutes)

### I want to understand changes
→ Read: **KAGGLE_UPDATES_SUMMARY.md** (10 minutes)

## 🎓 Key Concepts

### Two Phases
- **Phase 1**: Train baseline random model (8-10h)
- **Phase 2**: Iteratively improve model (8-12h)

### Progress Tracking
- Saves state after each task completes
- Allows automatic resumption after timeouts
- Prevents duplicate work

### Automatic Resumption
- Script detects completed tasks
- Skips them on re-run
- Picks up from last incomplete task

## 🐛 Troubleshooting

### "ModuleNotFoundError: gymnasium"
→ Automatically installed. If fails, restart kernel.

### "CUDA out of memory"
→ Reduce batch_size (100→50 for VAE, 25→10 for MDRNN)

### "Timeout interrupted task"
→ Just run again. Script will resume automatically.

### "Want to restart from scratch"
→ Delete `kaggle_progress.json` and re-run

### "Want to run Phase 2 only"
→ Run Phase 1 first (or set RUN_PHASE="2" if Phase 1 done)

## 🎯 Success Checklist

- [ ] Read KAGGLE_QUICK_START.md (2 min)
- [ ] Copy epls_kaggle.py or epls_kaggle.ipynb to Kaggle
- [ ] Run the script/notebook
- [ ] Wait for Phase 1 to complete (~10h)
- [ ] If timeout, run again (automatic resume)
- [ ] Phase 2 completes automatically (~12h)
- [ ] Check results in tests_custom/planning_test_results/
- [ ] Verify scores: Random ~356, Iterative ~708

## 📊 Files Generated After Run

```
/kaggle/working/WorldModelPlanning/
├── config.json                          (active config)
├── kaggle_progress.json                 (progress tracker)
├── mdrnn/checkpoints/
│   ├── World_Model_Random_mdrnn_best.tar
│   └── World_Model_Iter_A_mdrnn_best.tar
├── vae/checkpoints/
│   ├── World_Model_Random_vae_best.tar
│   └── World_Model_Iter_A_vae_best.tar
├── tests_custom/planning_test_results/
│   ├── CarRacing-v2_RMHC_World_Model_Random_*.pickle
│   └── CarRacing-v2_RMHC_World_Model_Iter_A_*.pickle
└── data_random_raw/                     (training data)
```

## 🚀 Next Steps

1. **Choose format**: Script (.py) or Notebook (.ipynb)?
2. **Read quick start**: 2 minutes max
3. **Set up on Kaggle**: Copy file
4. **Run**: Let it do its thing
5. **Wait**: 20 hours total across sessions
6. **Celebrate**: You've trained a world model!

## 📞 Quick Reference

| Need | File |
|------|------|
| Quick start | KAGGLE_QUICK_START.md |
| Full guide | KAGGLE_NOTEBOOK_GUIDE.md |
| Timeouts | KAGGLE_RESUMPTION_GUIDE.md |
| Notebook help | NOTEBOOK_USER_GUIDE.md |
| Script help | See epls_kaggle.py comments |
| Comparison | SCRIPT_VS_NOTEBOOK.md |
| Local setup | EXECUTION_GUIDE.md |

---

**Everything you need is here. Pick a format and start!** 🚀

Last Updated: May 2024
