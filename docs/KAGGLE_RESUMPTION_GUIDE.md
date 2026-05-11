# Kaggle Resumption Guide - Progress Tracking

Comprehensive guide on how the progress tracking system works and how to handle Kaggle timeouts.

## 🔄 Automatic Progress Tracking

The `epls_kaggle.py` script maintains a `kaggle_progress.json` file that tracks:
- Current phase (1 or 2)
- Last completed stage
- List of all completed tasks
- Timestamp of last save

## 📋 Progress File Structure

### Initial State (First Run)
```json
{
  "phase": 1,
  "stage": null,
  "completed_tasks": [],
  "timestamp": null
}
```

### After Task Completion
```json
{
  "phase": 1,
  "stage": "1A_data_generated",
  "completed_tasks": [
    "phase1_generate_data"
  ],
  "timestamp": "2024-05-11 02:30:00"
}
```

### After Phase 1 Completion
```json
{
  "phase": 2,
  "stage": "phase1_complete",
  "completed_tasks": [
    "phase1_generate_data",
    "phase1_train_vae",
    "phase1_train_mdrnn",
    "phase1_benchmark"
  ],
  "timestamp": "2024-05-11 08:45:00"
}
```

### After Full Completion
```json
{
  "phase": 3,
  "stage": "phase2_complete",
  "completed_tasks": [
    "phase1_generate_data",
    "phase1_train_vae",
    "phase1_train_mdrnn",
    "phase1_benchmark",
    "phase2_setup_checkpoint",
    "phase2_iterative_train",
    "phase2_benchmark"
  ],
  "timestamp": "2024-05-11 18:30:00"
}
```

## 🎯 Task Flow Diagram

```
Kaggle Session 1: TIMEOUT at 9 hours
═════════════════════════════════════
[START]
  ↓
[Load kaggle_progress.json]
  ├─ Phase: 1
  ├─ Stage: null
  └─ Completed: []
  ↓
[1A] Generate Data
  ├─ Duration: 2:30 min
  ├─ Save Progress: phase1_generate_data
  └─ Result: ✅
  ↓
[1B] Train VAE
  ├─ Duration: 1:15 min
  ├─ Save Progress: phase1_train_vae
  └─ Result: ✅
  ↓
[1C] Train MDRNN  ← STARTED
  ├─ Duration: 3:45 min (elapsed)
  ├─ Save Progress: SKIPPED (in progress)
  └─ Result: ⏱️ TIMEOUT
  
STATUS AT TIMEOUT: 7:30 min elapsed
Completed Tasks: 2
Last Saved Stage: 1B_vae_trained
Progress File: {phase: 1, completed_tasks: [1A, 1B]}

═════════════════════════════════════
Kaggle Session 2: RESUME
═════════════════════════════════════
[START]
  ↓
[Load kaggle_progress.json]
  ├─ Phase: 1
  ├─ Stage: 1B_vae_trained
  └─ Completed: [phase1_generate_data, phase1_train_vae]
  ↓
[1A] Check: is_task_done("phase1_generate_data")?
  └─ YES → Skip (print "✅ already done")
  ↓
[1B] Check: is_task_done("phase1_train_vae")?
  └─ YES → Skip (print "✅ already done")
  ↓
[1C] Check: is_task_done("phase1_train_mdrnn")?
  └─ NO → Execute
  ├─ Duration: 2:45 min (fresh start)
  ├─ Save Progress: phase1_train_mdrnn
  └─ Result: ✅
  ↓
[1D] Check: is_task_done("phase1_benchmark")?
  └─ NO → Execute
  ├─ Duration: 2:30 min
  ├─ Save Progress: phase1_benchmark
  └─ Result: ✅
  ↓
[Update Phase]
  ├─ Set: phase = 2
  ├─ Save Progress: phase1_complete
  └─ Result: ✅

STATUS AT END: 7:50 min elapsed
All Phase 1 Tasks Complete: YES
Progress File: {phase: 2, completed_tasks: [1A, 1B, 1C, 1D]}
```

## 🔍 How Each Task Checks Completion

### Data Generation Check
```python
if not is_task_done("phase1_generate_data", progress):
    # Execute task
else:
    print("✅ Task 1A: Data generation (already done)")
    # Skip
```

### Model Training Check
```python
if not is_task_done("phase1_train_vae", progress):
    if checkpoint_exists("World_Model_Random", "vae"):
        print("   ℹ️  VAE checkpoint exists, skipping training")
    else:
        # Execute training
    # Mark as done
else:
    print("✅ Task 1B: VAE training (already done)")
    # Skip
```

### Benchmark Check
```python
if not is_task_done("phase1_benchmark", progress):
    # Execute 100 trials (may timeout again)
    # Mark as done when complete
else:
    print("✅ Task 1D: Benchmarking (already done)")
    # Skip
```

## ⏳ Timeline Examples

### Scenario 1: Quick Run (No Timeout)
```
Session 1: 7 hours 50 minutes
├─ 1A Data Generation:  2:30 (00:00 - 02:30)
├─ 1B VAE Training:     1:15 (02:30 - 03:45)
├─ 1C MDRNN Training:   2:45 (03:45 - 06:30)
└─ 1D Benchmarking:     2:00 (06:30 - 08:30)
→ Phase 1 COMPLETE
→ Next: Run Phase 2
```

### Scenario 2: Timeout During MDRNN
```
Session 1: 9:00 (timeout)
├─ 1A Data Generation:  2:30 ✅ (saved)
├─ 1B VAE Training:     1:15 ✅ (saved)
├─ 1C MDRNN Training:   (5:15 elapsed before timeout)
│   └─ ❌ NOT SAVED (in progress when timeout)
└─ 1D Not started

Session 2: Resume
├─ 1A Data Gen:    SKIP (2 sec)
├─ 1B VAE:         SKIP (2 sec)
├─ 1C MDRNN:       RUN (full 2:45 from start)
└─ 1D Benchmark:   RUN (2:00)
→ Total time: 5:00 min
→ Phase 1 COMPLETE
```

### Scenario 3: Timeout During Benchmark
```
Session 1: 9:00 (timeout)
├─ 1A Data:    2:30 ✅
├─ 1B VAE:     1:15 ✅
├─ 1C MDRNN:   2:45 ✅
└─ 1D Bench:   (2:30 elapsed on 100 trials, completed 30%)
   └─ ❌ NOT SAVED

Session 2: Resume
├─ 1A-1C: SKIP (all done)
└─ 1D Bench: RUN from scratch (re-run all 100 trials)
   → TIP: Can reduce trials in config to save time
```

## 🛠️ Manual Progress Control

### Check Current Progress
```python
# In Kaggle notebook, add cell:
import json

if os.path.exists("kaggle_progress.json"):
    with open("kaggle_progress.json") as f:
        progress = json.load(f)
    print(json.dumps(progress, indent=2))
else:
    print("No progress file yet - first run")
```

### Force Skip to Phase 2
```python
# In Kaggle notebook, before running script:
import json

progress = {
    "phase": 2,
    "stage": "phase1_complete",
    "completed_tasks": [
        "phase1_generate_data",
        "phase1_train_vae",
        "phase1_train_mdrnn",
        "phase1_benchmark"
    ],
    "timestamp": "2024-05-11 12:00:00"
}

with open("kaggle_progress.json", "w") as f:
    json.dump(progress, f, indent=2)

# Then run:
!python epls_kaggle.py  # Jumps to Phase 2
```

### Reset to Start Over
```python
# Delete progress file to start fresh
import os
if os.path.exists("kaggle_progress.json"):
    os.remove("kaggle_progress.json")

# Then run:
!python epls_kaggle.py  # Starts from beginning
```

### Mark Task as Done Manually
```python
# If a task completed but progress wasn't saved:
import json

with open("kaggle_progress.json") as f:
    progress = json.load(f)

progress["completed_tasks"].append("phase1_train_vae")
progress["stage"] = "1B_vae_trained"

with open("kaggle_progress.json", "w") as f:
    json.dump(progress, f, indent=2)
```

## 📊 Monitoring Script Output

### What You'll See at Startup

**First Run:**
```
✅ Helpers ready
📊 Progress: Phase 1, Last stage: None
   Completed tasks: 0
```

**After Resume:**
```
✅ Helpers ready
📊 Progress: Phase 1, Last stage: 1B_vae_trained
   Completed tasks: 2
```

### What You'll See During Execution

**Starting a Task:**
```
⏳ Task 1A: Generating random rollouts...
   Found 0/5000. Generating 5000 more...
[Training/generation output...]
  📝 Progress saved: Phase 1, Stage: 1A_data_generated
```

**Skipping a Task:**
```
✅ Task 1B: VAE training (already done)
```

**Completing a Phase:**
```
========================================
✅ PHASE 1 COMPLETE
========================================
Random Model results in: tests_custom/planning_test_results/
```

## 🎯 Final Summary Output

After all tasks complete:
```
✨ KAGGLE EXECUTION SUMMARY
========================================

📊 Progress Report:
   Current Phase: 3
   Last Stage: phase2_complete
   Completed Tasks (7):
     ✓ phase1_generate_data
     ✓ phase1_train_vae
     ✓ phase1_train_mdrnn
     ✓ phase1_benchmark
     ✓ phase2_setup_checkpoint
     ✓ phase2_iterative_train
     ✓ phase2_benchmark

⏳ Remaining tasks (0):

✅ All Phase 3 tasks completed!

📁 Results Locations:
   - Model checkpoints: mdrnn/checkpoints/
   - Benchmarks: tests_custom/planning_test_results/
   - Iteration stats: mdrnn/iteration_stats/
   - Progress tracker: kaggle_progress.json

⚙️  Resumption Instructions:
   If Kaggle times out:
   1. Run again: EPLS_RUN_PHASE=all python epls_kaggle.py
   2. Script automatically resumes from last completed task
   3. Check kaggle_progress.json for detailed progress
```

## ⚠️ Troubleshooting

### Problem: Progress file corrupted
**Solution:**
```python
import os
os.remove("kaggle_progress.json")
# Run script again - will start fresh
```

### Problem: Script thinks task is done but it's not
**Solution:**
1. Check if checkpoint files exist:
```python
import os
print(os.path.exists("mdrnn/checkpoints/World_Model_Random_mdrnn_best.tar"))
```
2. If checkpoint exists, training won't run (by design)
3. Delete checkpoint if you want to retrain:
```python
import os
os.remove("mdrnn/checkpoints/World_Model_Random_mdrnn_best.tar")
```

### Problem: Want to restart a specific phase
**Solution:**
```python
import json

# Clear Phase 2 only
with open("kaggle_progress.json") as f:
    progress = json.load(f)

# Remove Phase 2 tasks
progress["completed_tasks"] = [t for t in progress["completed_tasks"] if "phase2" not in t]
progress["phase"] = 2
progress["stage"] = None

with open("kaggle_progress.json", "w") as f:
    json.dump(progress, f, indent=2)
```

## 🔐 Data Persistence

All important data persists between Kaggle sessions:
- ✅ Model checkpoints: `mdrnn/checkpoints/`, `vae/checkpoints/`
- ✅ Training data: `data_random_raw/`, `data_iterative/`
- ✅ Benchmark results: `tests_custom/planning_test_results/`
- ✅ Progress file: `kaggle_progress.json`
- ⚠️ Notebooks are ephemeral (save your code separately)

## 📈 Expected Time Allocations

### Phase 1 (8-10 hours total)
- Task 1A: 2-3 hours
- Task 1B: 1-2 hours
- Task 1C: 2-3 hours
- Task 1D: 2-3 hours

### Phase 2 (8-12 hours total)
- Task 2A: <5 minutes (skip if using linked data)
- Task 2B: 6-10 hours (5 iterations × 1-2 hours each)
- Task 2C: 2-3 hours (100 trials)

**Total: 16-22 hours** (may require multiple Kaggle sessions)

---

**Key Insight**: The progress tracking system is designed so that **no task ever runs twice**. Each completed task is recorded, and resuming skips it entirely.
