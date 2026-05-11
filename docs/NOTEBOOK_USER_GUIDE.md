# Jupyter Notebook User Guide - epls_kaggle.ipynb

Complete guide for using the `epls_kaggle.ipynb` notebook on Kaggle.

## 📖 What's in the Notebook?

The notebook is divided into **13 logical cells**:

### Setup & Configuration (Cells 1-5)
1. **Install Dependencies** - Installs gymnasium, torch, and other packages
2. **Helper Functions** - Defines utility functions (config, checkpoints, etc.)
3. **Project Setup** - Clones/updates GitHub repository
4. **Link Previous Outputs** - Reuses outputs from previous Kaggle runs
5. **Progress Tracking** - Initializes progress tracking and Xvfb display

### Phase 1: Random Baseline (Cells 6-9)
6. **Task 1A** - Generate 5,000 random rollouts
7. **Task 1B** - Train VAE (20 epochs)
8. **Task 1C** - Train MDRNN (60 epochs)
9. **Task 1D** - Benchmark with 100 trials

### Phase 2: Iterative Refinement (Cells 10-12)
10. **Task 2A** - Setup iterative checkpoint
11. **Task 2B** - Run iterative training (5 iterations)
12. **Task 2C** - Benchmark with 100 trials

### Results (Cell 13)
13. **Final Summary** - Progress report and results

## 🚀 Usage

### Option 1: Run All Cells (Default)
```python
# Select Kernel → "Run All" or Shift+Ctrl+Enter
# Or click the "Run All" button in toolbar
```

### Option 2: Run Specific Phases

**Run Phase 1 only:**
```python
# Before running, set:
RUN_PHASE = "1"

# Then run all cells (skips Phase 2)
```

**Run Phase 2 only:**
```python
# Before running, set:
RUN_PHASE = "2"

# Then run all cells (skips Phase 1)
```

### Option 3: Run Cells Sequentially

Click **Shift+Enter** on each cell to run it in order. Best for:
- Debugging issues
- Monitoring progress in detail
- Stopping at specific points

## ⏱️ Timeline

### Phase 1 (8-10 hours)
- Cell 6: 2-3 hours (data generation)
- Cell 7: 1-2 hours (VAE training)
- Cell 8: 2-3 hours (MDRNN training)
- Cell 9: 2-3 hours (benchmarking)

### Phase 2 (8-12 hours)
- Cell 10: <5 minutes (checkpoint setup)
- Cell 11: 6-10 hours (iterative training)
- Cell 12: 2-3 hours (benchmarking)

## 🔄 Handling Timeouts

If Kaggle times out (9-hour limit):

1. **Cell 13 shows progress** - Check which tasks completed
2. **Progress file saved** - `kaggle_progress.json` tracks state
3. **Just run again** - Jupyter will auto-skip completed cells
4. **Notebook remembers progress** - No manual intervention needed

**Example:**
```
Run 1: Cells 1-7 complete, timeout during Cell 8
       → kaggle_progress.json shows "phase1_train_vae" done

Run 2: Cells 1-7 skip (cached), Cell 8 runs fresh
       → Continues from where it stopped
```

## 📊 Monitoring Progress

### During Execution
Watch the output in real-time as each cell runs. Progress is printed:
```
⏳ Task 1A: Generating random rollouts...
   Found 2500/5000. Generating 2500 more...
[Training output...]
  📝 Progress saved: Phase 1, Stage: 1A_data_generated
```

### Between Sessions
Check `kaggle_progress.json` in the working directory:
```python
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

### Final Report
Cell 13 automatically generates a summary:
```
📊 Progress Report:
   Current Phase: 1
   Last Stage: 1B_vae_trained
   Completed Tasks (2):
     ✓ phase1_generate_data
     ✓ phase1_train_vae

⏳ Remaining tasks (2):
     ⋯ phase1_train_mdrnn
     ⋯ phase1_benchmark
```

## 🔧 Customization

### Modify Training Parameters

Edit the `set_config()` calls in each cell:

**Reduce rollouts (Phase 1, Cell 6):**
```python
set_config({
    ...
    "data_generator": {
        "rollouts": 1000,  # Changed from 5000
        ...
    }
})
```

**Increase VAE epochs (Cell 7):**
```python
"vae_trainer": {
    "max_epochs": 50,  # Changed from 20
    ...
}
```

**Reduce benchmark trials (Cell 9):**
```python
"test_suite": {
    "trials": 50,  # Changed from 100
    ...
}
```

### Use Different Planning Agent

Change in Cell 9 and Cell 12:
```python
set_config({
    ...
    "planning": {
        "planning_agent": "RHEA",  # Options: RMHC, RHEA, MCTS
        ...
    }
})
```

## 📁 Output Locations

After notebook completes, find results in:
```
/kaggle/working/WorldModelPlanning/
├── config.json
├── kaggle_progress.json
├── mdrnn/checkpoints/
│   ├── World_Model_Random_mdrnn_best.tar
│   └── World_Model_Iter_A_mdrnn_best.tar
├── vae/checkpoints/
│   ├── World_Model_Random_vae_best.tar
│   └── World_Model_Iter_A_vae_best.tar
├── data_iterative/
├── data_random_raw/
└── tests_custom/planning_test_results/
    ├── CarRacing-v2_RMHC_World_Model_Random_*.pickle
    └── CarRacing-v2_RMHC_World_Model_Iter_A_*.pickle
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Cell 1 installs dependencies. Restart kernel if still failing.

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in Cell 7 and Cell 8:
```python
"vae_trainer": {"batch_size": 50},      # from 100
"mdrnn_trainer": {"batch_size": 10},    # from 25
```

### Issue: Script says phase not done but I see results
**Solution**: Check `kaggle_progress.json`. If missing task:
```python
import json
with open("kaggle_progress.json") as f:
    p = json.load(f)
p["completed_tasks"].append("phase1_train_vae")  # Add manually
with open("kaggle_progress.json", "w") as f:
    json.dump(p, f, indent=2)
```

### Issue: Want to restart from beginning
**Solution**: Delete progress file and re-run:
```python
import os
if os.path.exists("kaggle_progress.json"):
    os.remove("kaggle_progress.json")
# Then run notebook again
```

## 💡 Pro Tips

### 1. Run Cells with Keyboard Shortcuts
- **Shift+Enter** - Run current cell
- **Ctrl+Enter** - Run cell (stay in place)
- **Alt+Enter** - Run cell and insert new cell below

### 2. Save Notebook Regularly
Kaggle auto-saves, but manually save with **Ctrl+S** to be safe.

### 3. Download Results
After Phase 1 completes, download `mdrnn/` and `vae/` checkpoints as dataset for Phase 2 in new notebook.

### 4. Link Previous Outputs
In next Kaggle notebook, add previous outputs as dataset input:
1. Click "Add Input" button
2. Select dataset with checkpoint files
3. Run Cell 4 - automatically links files

### 5. Monitor with Markdown
Add markdown cells between code cells to document progress:
```markdown
## Iteration 3 Complete
- VAE Loss: 145.2
- MDRNN Loss: 4.5
- Next: Run benchmarks
```

## 📊 Expected Outputs

### Phase 1 Completion
```
✅ PHASE 1 COMPLETE
Random Model results in: tests_custom/planning_test_results/

Expected Scores:
- Random Model: ~356 ± 177 avg reward
```

### Phase 2 Completion
```
✅ PHASE 2 COMPLETE
Iterative Model results in: tests_custom/planning_test_results/

Expected Scores:
- Iterative Model: ~708 ± 195 avg reward
```

## 🔐 Data Persistence

All outputs persist between Kaggle sessions:
- ✅ Model checkpoints (always preserved)
- ✅ Training data (always preserved)
- ✅ Benchmark results (always preserved)
- ✅ Progress file (always preserved)
- ⚠️ Notebook itself (save as version to preserve)

## 📚 Related Documentation

- **KAGGLE_QUICK_START.md** - 2-minute setup
- **KAGGLE_NOTEBOOK_GUIDE.md** - Complete guide
- **KAGGLE_RESUMPTION_GUIDE.md** - Progress tracking details
- **EXECUTION_GUIDE.md** - Local execution (same code)

## 🎓 Key Concepts

### What Each Phase Does

**Phase 1:**
- Trains baseline models on random data
- Evaluates baseline performance
- Provides foundation for iterative refinement

**Phase 2:**
- Uses planning agent to collect better data
- Iteratively refines MDRNN with collected data
- Improves performance through self-play

### Progress Tracking

Each task updates `kaggle_progress.json` after completion:
- Prevents duplicate work
- Enables automatic resumption
- Provides audit trail

### Timeout Recovery

If Kaggle kills notebook at 9 hours:
1. Progress saved to file
2. Notebook can be re-run
3. Completed tasks auto-skipped
4. Execution resumes seamlessly

---

**That's it!** Run the notebook and let it handle the rest. 🚀
