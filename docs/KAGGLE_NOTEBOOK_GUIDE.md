# Kaggle Notebook Guide - EPLS World Model

Complete guide for running the **Evolutionary Planning in Latent Space** project on Kaggle with automatic progress tracking and resumption.

## 🎯 Quick Start

### Option 1: Use Pre-made Kaggle Notebook
1. Open: https://kaggle.com/code/your_username/epls-world-model
2. Click **"Run All"** button in top-right
3. Script automatically handles timeouts and resumes

### Option 2: Upload Python Script
1. Create new Kaggle Notebook (Python)
2. Copy code from `epls_kaggle.py`
3. Paste into first cell
4. Click **Run**

## 📊 Workflow Overview

The script runs a **2-phase pipeline** with automatic progress tracking:

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Random Baseline (8-10 hours)                      │
├─────────────────────────────────────────────────────────────┤
│  1A. Generate 5,000 random rollouts  (2-3 hours)           │
│  1B. Train VAE                        (1-2 hours)           │
│  1C. Train MDRNN                      (2-3 hours)           │
│  1D. Benchmark (100 trials)           (2-3 hours)           │
│  → Target: ~356 avg reward                                  │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Iterative Refinement (8-12 hours)                │
├─────────────────────────────────────────────────────────────┤
│  2A. Setup iterative checkpoint       (<5 min)              │
│  2B. Run 5 iterations of refinement   (6-10 hours)          │
│  2C. Benchmark (100 trials)           (2-3 hours)           │
│  → Target: ~708 avg reward                                  │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ Running Phases

### Run Everything (Default)
```python
# In Kaggle notebook cell:
!python epls_kaggle.py
```

### Run Only Phase 1
```python
import os
os.environ["EPLS_RUN_PHASE"] = "1"
!python epls_kaggle.py
```

### Run Only Phase 2
```python
import os
os.environ["EPLS_RUN_PHASE"] = "2"
!python epls_kaggle.py
```

## 🔄 Handling Kaggle Timeouts

Kaggle notebooks have **9-hour session limits**. The script automatically handles resumption:

### What Happens on Timeout

1. **Session ends** at 9-hour mark
2. **Script was interrupted** mid-task
3. **Progress file saved** before last task
4. **Next run skips completed tasks**

### Example Timeline

```
Run 1 (Session 1):
  [0:00] Start
  [2:30] ✅ 1A: Data generation complete
  [3:45] ✅ 1B: VAE training complete
  [7:30] 🔄 Starting MDRNN training...
  [9:00] ⏱️ Kaggle timeout - session ended

kaggle_progress.json contains:
  {
    "phase": 1,
    "stage": "1B_vae_trained",
    "completed_tasks": ["phase1_generate_data", "phase1_train_vae"],
    "timestamp": "2024-05-11 03:45:00"
  }

Run 2 (Session 2):
  [0:00] Start - reads progress file
  [0:10] ✅ 1A: Data generation (skip)
  [0:15] ✅ 1B: VAE training (skip)
  [0:20] 🚀 Resume 1C: MDRNN training (from scratch, ~2 hours)
  [2:45] ✅ 1C: MDRNN complete
  [5:15] ✅ 1D: Benchmarking complete
```

## 📈 Monitoring Progress

### Check Progress During Run

The script prints progress at startup:
```
✅ Helpers ready
📊 Progress: Phase 1, Last stage: 1B_vae_trained
   Completed tasks: 2
```

### View Progress File

After each task completes, check `kaggle_progress.json`:
```json
{
  "phase": 1,
  "stage": "1C_mdrnn_trained",
  "completed_tasks": [
    "phase1_generate_data",
    "phase1_train_vae",
    "phase1_train_mdrnn"
  ],
  "timestamp": "2024-05-11 06:30:00"
}
```

## 💾 Output Structure

After successful runs, find results in:

```
/kaggle/working/WorldModelPlanning/
├── config.json                          # Current config
├── kaggle_progress.json                 # Progress tracker
├── mdrnn/
│   ├── checkpoints/
│   │   ├── World_Model_Random_mdrnn_best.tar      # Phase 1
│   │   ├── World_Model_Iter_A_mdrnn_best.tar      # Phase 2
│   │   └── backups/
│   │       └── iterative_World_Model_Iter_A_*.tar # Each iteration
│   └── iteration_stats/
│       └── iterative_stats_World_Model_Iter_A.pickle
├── vae/
│   └── checkpoints/
│       ├── World_Model_Random_vae_best.tar
│       └── World_Model_Iter_A_vae_best.tar
├── tests_custom/planning_test_results/
│   ├── CarRacing-v2_RMHC_World_Model_Random_...pickle
│   └── CarRacing-v2_RMHC_World_Model_Iter_A_...pickle
└── data_iterative/
    ├── iteration_0/
    ├── iteration_1/
    └── ...
```

## 🎯 Expected Results

### Phase 1 (Random Model)
```
Epoch 20/20 (VAE):     Loss: ~150-200
Epoch 60/60 (MDRNN):   Loss: ~3-5
Benchmark (100 runs):  Avg Reward: 300-400 (Target: ~356)
```

### Phase 2 (Iterative Model)
```
Iteration 1: Avg Reward: 380-420
Iteration 2: Avg Reward: 420-480
Iteration 3: Avg Reward: 480-550
Iteration 4: Avg Reward: 550-650
Iteration 5: Avg Reward: 650-750
Final Benchmark: Avg Reward: 650-750 (Target: ~708)
```

## 🔧 Customization

### Reduce Computation Time

Edit phase configuration directly in script:

```python
# Reduce from 5000 to 1000 rollouts
"data_generator": {
    "rollouts": 1000,  # Changed from 5000
    ...
}

# Reduce from 100 to 50 trials
"test_suite": {
    "trials": 50,  # Changed from 100
    ...
}

# Reduce iterations from 5 to 3
"iterative_trainer": {
    "num_iterations": 3,  # Changed from 5
    ...
}
```

### Use Pre-trained Models

If you have previous Kaggle outputs saved as a dataset:

```python
# Dataset linked as /kaggle/input/previous-phase-output/
# Script automatically links:
# - data_random_raw/
# - mdrnn/checkpoints/
# - vae/checkpoints/
```

Just run the script—it will skip data generation and training if checkpoints exist!

### Increase Training Duration

To reach higher scores:

```python
# Increase MDRNN training
"mdrnn_trainer": {
    "max_epochs": 100,  # Changed from 60
    ...
}

# Increase iterations
"iterative_trainer": {
    "num_iterations": 10,  # Changed from 5
    ...
}

# Increase benchmark trials
"test_suite": {
    "trials": 500,  # Changed from 100
    ...
}
```

## ⚠️ Common Issues

### Issue 1: "ModuleNotFoundError: gymnasium"
**Solution**: Dependencies automatically installed at script start. If still failing, restart kernel.

### Issue 2: "CUDA out of memory"
**Solution**: Reduce batch size in config
```python
"vae_trainer": {"batch_size": 50},      # from 100
"mdrnn_trainer": {"batch_size": 10},    # from 25
```

### Issue 3: "File not found: config.json"
**Solution**: Script creates config.json automatically. If missing, check working directory:
```python
import os
print(os.getcwd())  # Should be /kaggle/working/WorldModelPlanning
```

### Issue 4: Previous outputs not linking
**Solution**: Make sure dataset is properly added in Kaggle notebook settings:
1. Click **"Add Input"** button
2. Select dataset with previous phase outputs
3. Restart kernel before running script

## 📊 Comparing Results

After both phases complete, view TensorBoard logs in the `/utility/logging/tensorboard_runs/` directory structure.

To extract scores from pickle files:

```python
import pickle

# Load Phase 1 results
with open('tests_custom/planning_test_results/CarRacing-v2_RMHC_World_Model_Random_*.pickle', 'rb') as f:
    results_p1 = pickle.load(f)

# Load Phase 2 results
with open('tests_custom/planning_test_results/CarRacing-v2_RMHC_World_Model_Iter_A_*.pickle', 'rb') as f:
    results_p2 = pickle.load(f)

print(f"Phase 1 (Random): {results_p1['agent_params']}")
print(f"Phase 2 (Iterative): {results_p2['agent_params']}")
```

## 📝 Progress Tracking Details

### Task States

Each task can be:
- **✅ Done**: Marked in `completed_tasks` list, skipped on resume
- **🔄 In Progress**: Being executed, progress file updated after completion
- **⏳ Pending**: Not yet started, will run on this session

### Force Restart from Beginning

To force a fresh start (useful for debugging):

```python
import os
if os.path.exists("kaggle_progress.json"):
    os.remove("kaggle_progress.json")
# Then run the script
!python epls_kaggle.py
```

### Manual Progress Update

If script crashes during a task and progress file wasn't saved:

```python
import json

progress = {
    "phase": 1,
    "stage": "manual_reset",
    "completed_tasks": ["phase1_generate_data", "phase1_train_vae"],
    "timestamp": "2024-05-11 12:00:00"
}

with open("kaggle_progress.json", "w") as f:
    json.dump(progress, f, indent=2)
```

## 🚀 Optimization Tips

1. **Use GPU for training**: Kaggle provides free Tesla T4/P100 GPU access
2. **Link previous outputs**: Saves re-running data generation/training
3. **Monitor TensorBoard**: View loss curves in real-time
4. **Set reasonable timeouts**: Phase 1 takes ~8-10 hours, Phase 2 takes ~8-12 hours
5. **Use replay buffer**: Enabled by default, prevents catastrophic forgetting

## 📚 Related Files

- **epls_kaggle.py** - Main Kaggle script (this file)
- **main.py** - Local entry point (used by script)
- **config.json** - Configuration (auto-generated)
- **EXECUTION_GUIDE.md** - Detailed local execution guide
- **README.md** - Original project documentation

## 💡 Pro Tips

1. **Download outputs as dataset**: After Phase 1 completes, output the `mdrnn/` and `vae/` folders as a Kaggle dataset. Then use it as input for Phase 2 in a new notebook.

2. **Parallelize phases**: Run Phase 1 in one notebook, Phase 2 in another simultaneously (if you have multiple notebooks quota).

3. **Archive results**: Download `tests_custom/planning_test_results/` pickle files for analysis.

4. **Share progress**: Commit `kaggle_progress.json` to a version to show collaborators what stage you're at.

## 🎓 Paper Reference

- **Title**: Evolutionary Planning in Latent Space  
- **Authors**: Olesen et al., 2020
- **Paper**: https://arxiv.org/abs/2011.11293
- **World Models**: https://doi.org/10.5281/zenodo.1207631

---

**Last Updated**: May 2024  
**Compatible With**: Python 3.9+, PyTorch 2.x, Gymnasium 1.1.1+, Kaggle GPU (Tesla T4/P100)
