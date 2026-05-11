# Execution Guide: World Model Planning with Evolutionary Algorithms

This guide covers running the **modernized** EPLS (Evolutionary Planning in Latent Space) project after the recent updates to gymnasium 1.1.1, PyTorch 2.x, and Python 3.9+.

## 📋 Prerequisites & Environment Setup

### 1. Python Version
- **Minimum**: Python 3.9
- **Recommended**: Python 3.10 or 3.11
- **Test with**: `python --version`

### 2. Install Dependencies

#### Option A: Using conda + pip (Recommended)
```bash
# Create conda environment
conda create -n world_model python=3.9
conda activate world_model

# Install torch and torchvision via conda (faster)
conda install pytorch::pytorch torchvision -c pytorch
conda install pytorch-cuda=11.8 -c pytorch -c conda-forge  # Optional: for GPU support

# Install gymnasium and other dependencies via pip
pip install gymnasium[box2d]>=0.29.0
pip install numpy>=1.26.0
pip install matplotlib
pip install tensorboard>=2.12
pip install tqdm
pip install colorama
pip install dill
```

#### Option B: Using pip only
```bash
pip install torch>=2.0 torchvision>=0.15
pip install gymnasium[box2d]>=0.29.0
pip install numpy>=1.26.0
pip install matplotlib tensorboard tqdm colorama dill
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium {gymnasium.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

**Expected Output:**
```
PyTorch 2.x.x
Gymnasium 1.x.x
NumPy 1.26.x or higher
```

## 🎯 Project Structure Overview

```
WorldModelPlanning/
├── config.json                      # Main configuration file
├── main.py                          # Entry point
├── vae/                            # VAE model and training
│   ├── vae.py                      # ConvVAE architecture
│   ├── vae_trainer.py              # VAE training loop
│   └── checkpoints/                # Trained VAE models
├── mdrnn/                          # MDRNN model and training
│   ├── mdrnn.py                    # MDRNN + MDN architecture
│   ├── mdrnn_trainer.py            # MDRNN training loop
│   ├── iterative_trainer.py        # Iterative refinement
│   ├── learning.py                 # EarlyStopping utility
│   ├── checkpoints/                # Trained MDRNN models
│   └── iteration_stats/            # Iteration results
├── planning/                       # Planning algorithms
│   ├── simulation/                 # RHEA, RMHC, MCTS implementations
│   └── simulated_planning_controller.py  # Dream environment
├── environment/                    # Environment wrappers
│   ├── carracing/                  # Car Racing specifics
│   └── base_environment.py         # Abstract base
├── tests_custom/                   # Benchmarking and testing
│   ├── car_racing/                 # Car Racing test suites
│   └── other/                      # Other environment tests
├── utility/                        # Utilities
│   ├── rollout_handling/           # Data generation
│   ├── logging/                    # TensorBoard logging
│   └── visualizer.py               # Visualization tools
├── data/                           # Training data (auto-created)
├── data_test/                      # Test data (auto-created)
├── data_iterative/                 # Iterative training data (auto-created)
└── pyproject.toml                  # Dependencies and project metadata
```

## ⚙️ Configuration File Basics (`config.json`)

The `config.json` file controls all aspects of the pipeline. Key sections:

### Model Selection
```json
{
  "experiment_name": "World_Model_Random",
  "latent_size": 64,
  "game": "CarRacing-v2"
}
```

### Feature Flags (Run Stages)
```json
{
  "is_generate_data": false,           // Generate rollouts
  "is_train_vae": false,               // Train VAE
  "is_train_mdrnn": false,             // Train MDRNN
  "is_iterative_train_mdrnn": false,   // Iterative refinement
  "is_play": false,                    // Play live game
  "is_dream_play": false,              // Play in dream
  "is_manual_control": false,          // Manual control
  "is_ntbea_param_tune": false,        // Parameter tuning
  "is_run_planning_tests": false       // Benchmark planning
}
```

### Directory Paths
```json
{
  "vae_dir": "vae",
  "mdrnn_dir": "mdrnn",
  "data_dir": "data",
  "test_data_dir": "data_test",
  "data_generator": {
    "data_output_dir": "data_random_raw"
  }
}
```

## 🚀 Step-by-Step Execution Pipeline

### Phase 1: Generate Training Data (Optional)

**When to use**: When training a new VAE or MDRNN from scratch.

```bash
# 1. Edit config.json
{
  "is_generate_data": true,
  "data_generator": {
    "rollouts": 10000,               # Number of rollouts
    "sequence_length": 501,          # Steps per rollout
    "data_prefix": "random",
    "car_racing": {
      "is_ha_agent_driver": false    # true for expert, false for random
    }
  }
}

# 2. Run generation
python main.py

# Expected output:
# - Creates data_random_raw/ directory
# - Generates ~10GB of data (10k rollouts × 501 frames)
# - Estimated time: 1-4 hours on CPU
```

**Output files**:
- `data_random_raw/train.npy` - Training frames (~8GB)
- `data_random_raw/train_rewards.npy` - Reward sequences
- `data_random_raw/train_actions.npy` - Action sequences

### Phase 2: Train VAE

**Prerequisites**: Training data from Phase 1

```bash
# 1. Edit config.json
{
  "is_generate_data": false,
  "is_train_vae": true,
  "experiment_name": "MY_VAE_NAME",
  "latent_size": 64,
  "vae_trainer": {
    "max_epochs": 50,
    "batch_size": 100,
    "learning_rate": 0.0001,
    "train_buffer_size": 50,
    "test_buffer_size": 50,
    "is_continue_model": false  # true to resume from checkpoint
  }
}

# 2. Run training
python main.py

# Expected output in terminal:
# Epoch 1/50: Loss: 1234.5 | KL: 45.2 | Recon: 1189.3
# [validation every epoch]
```

**Monitoring with TensorBoard**:
```bash
tensorboard --logdir utility/logging/tensorboard_runs --port 6006
# Open browser: http://localhost:6006
```

**Output files**:
- `vae/checkpoints/MY_VAE_NAME_vae_best.tar` - Best model (checkpoint)
- `vae/checkpoints/MY_VAE_NAME_vae_checkpoint.tar` - Latest checkpoint

**Typical Training Time**: 2-4 hours on GPU, 8-16 hours on CPU

### Phase 3: Train MDRNN

**Prerequisites**: Trained VAE from Phase 2

```bash
# 1. Edit config.json
{
  "is_train_vae": false,
  "is_train_mdrnn": true,
  "experiment_name": "MY_VAE_NAME",  # Must match VAE name
  "latent_size": 64,
  "mdrnn": {
    "hidden_units": 512,
    "num_gaussians": 5
  },
  "mdrnn_trainer": {
    "max_epochs": 60,
    "batch_size": 25,
    "learning_rate": 0.001,
    "sequence_length": 500,
    "gradient_clip": 1.0,
    "is_continue_model": false,
    "early_stop_after_n_bad_epochs": 5,
    "ReduceLROnPlateau": {
      "mode": "min",
      "factor": 0.5,
      "patience": 2
    }
  }
}

# 2. Run training
python main.py

# Expected output in terminal:
# Epoch 1/60: Train Loss: 5.234 | Test Loss: 5.156
# [decreasing over epochs, should stop around epoch 20-30]
```

**Output files**:
- `mdrnn/checkpoints/MY_VAE_NAME_mdrnn_best.tar` - Best model
- `mdrnn/checkpoints/MY_VAE_NAME_mdrnn_checkpoint.tar` - Latest checkpoint

**Typical Training Time**: 3-6 hours on GPU, 12-24 hours on CPU

### Phase 4: Run Planning Tests (Benchmarking)

**Prerequisites**: Trained VAE + MDRNN OR pre-trained models

```bash
# 1. Edit config.json - Select models and agent
{
  "is_train_vae": false,
  "is_train_mdrnn": false,
  "experiment_name": "World_Model_Random",  # Pre-trained model
  "planning": {
    "planning_agent": "RMHC",  # Options: RMHC, RHEA, MCTS
    "random_mutation_hill_climb": {
      "horizon": 20,
      "max_generations": 12,
      "is_shift_buffer": true
    }
  },
  "test_suite": {
    "is_run_planning_tests": true,
    "is_reload_planning_session": false,
    "trials": 100,                    # Number of test runs
    "is_multithread_tests": false,
    "is_multithread_trials": false,   # Set true for parallel trials
    "is_logging": true
  }
}

# 2. Run tests
python main.py

# Expected output in terminal:
# Running: planning_whole_random_track
# Trial 1/100: Reward: 234.5 (Max: 250.2)
# [after ~2-4 hours]
# Average reward over 100 trials: 289.3
```

**Output files**:
- `tests_custom/planning_test_results/CarRacing-v2_RMHC_...pickle` - Results file
- TensorBoard logs in `utility/logging/tensorboard_runs/`

**View Results**:
```bash
tensorboard --logdir utility/logging/tensorboard_runs --port 6006
```

### Phase 5: Play in Real Environment (Live)

**Prerequisites**: Trained VAE + MDRNN

```bash
# 1. Edit config.json
{
  "experiment_name": "World_Model_Random",
  "planning": {
    "planning_agent": "RMHC",
    "random_mutation_hill_climb": {
      "horizon": 20,
      "max_generations": 12,
      "is_shift_buffer": true
    }
  },
  "is_play": true,
  "visualization": {
    "is_render": true
  }
}

# 2. Run
python main.py

# Expected output:
# Window opens showing Car Racing environment
# Agent controls car using planning algorithm
# Agent plays until done (500 steps or crash)
# Final reward printed to terminal
```

**Controls**:
- Window closes automatically when episode ends
- Check terminal for final reward

### Phase 6: Play in Dream (Simulated Environment)

**Prerequisites**: Trained VAE + MDRNN

```bash
# 1. Edit config.json
{
  "experiment_name": "World_Model_Random",
  "is_dream_play": true,
  "is_play": true,  # Usually enabled together
  "visualization": {
    "is_render_dream": true
  }
}

# 2. Run
python main.py

# Expected output:
# Shows MDRNN's predicted future (dream)
# Window displays sampled trajectory in latent space
# Much faster than real environment
```

### Phase 7: Iterative Training (Advanced)

**Prerequisites**: Trained MDRNN (as baseline)

Iteratively improves MDRNN by collecting rollouts from planning agent's policy.

```bash
# 1. Setup baseline model
#    Copy mdrnn/checkpoints/World_Model_Random_mdrnn_best.tar
#    to mdrnn/checkpoints/World_Model_Iter_A_mdrnn_best.tar

# 2. Edit config.json
{
  "experiment_name": "World_Model_Iter_A",
  "is_iterative_train_mdrnn": true,
  "iterative_trainer": {
    "num_iterations": 10,            # 10 refinement iterations
    "num_rollouts": 500,             # Per iteration
    "sequence_length": 250,
    "max_epochs": 10,
    "test_scenario": "planning_whole_random_track",
    "replay_buffer": {
      "is_replay_buffer": true,
      "max_buffer_size": 50000
    }
  },
  "planning": {
    "planning_agent": "RMHC"
  }
}

# 3. Run
python main.py

# Expected output:
# Iteration 1/10
#   - Generating 500 rollouts with planning agent...
#   - Training MDRNN with 500+previous rollouts...
#   - Testing performance...
# Iteration 2/10
#   ...
# Final model: mdrnn/checkpoints/iterative_World_Model_Iter_A_mdrnn_best.tar
```

**Output Structure**:
```
mdrnn/
├── checkpoints/
│   ├── World_Model_Iter_A_mdrnn_best.tar      # Baseline
│   ├── iterative_World_Model_Iter_A_mdrnn_best.tar    # Final
│   └── backups/
│       ├── iterative_World_Model_Iter_A_1.tar # Iteration 1
│       ├── iterative_World_Model_Iter_A_2.tar # Iteration 2
│       └── ...
└── iteration_stats/
    └── iterative_stats_World_Model_Iter_A.pickle
```

## 🔧 Pre-trained Models

The project includes several pre-trained models:

```
Model Name                        | Latent Size | MDRNN Hidden | Data Used     | Description
World_Model_Random                | 64          | 512          | 10k random    | Baseline random policy
World_Model_HaRandom              | 64          | 512          | Expert+random | Mixed training
iterative_World_Model_Iter_A      | 64          | 512          | 10k iterative | 10 iterations refined
```

**To use a pre-trained model**:
```json
{
  "is_generate_data": false,
  "is_train_vae": false,
  "is_train_mdrnn": false,
  "experiment_name": "World_Model_Random",  // Select which model
  "is_run_planning_tests": true              // Run benchmarks
}
```

## 📊 Monitoring Training Progress

### TensorBoard Dashboard

```bash
# Start TensorBoard server
tensorboard --logdir utility/logging/tensorboard_runs --port 6006

# Open browser: http://localhost:6006
```

**Available Metrics**:
- **Scalars**: Loss curves, learning rates, rewards
- **Images**: VAE reconstructions, dream visualizations
- **Histograms**: Weight distributions

### Log Files

Training logs are saved in human-readable format:
```
utility/logging/
├── tensorboard_runs/
│   ├── VAE_World_Model_Random_2024-05-11.../
│   ├── MDRNN_World_Model_Random_2024-05-11.../
│   └── ...
└── planning_logs/
    ├── World_Model_Random_RMHC_2024-05-11_...
    └── ...
```

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'gymnasium'"
**Solution**: 
```bash
pip install gymnasium[box2d]>=0.29.0
```

### Issue 2: "torch version mismatch" or "weights_only" parameter error
**Solution**: Update PyTorch
```bash
pip install --upgrade torch torchvision
```

### Issue 3: "Car Racing environment not found"
**Solution**: Reinstall gymnasium with Box2D support
```bash
pip uninstall gymnasium
pip install gymnasium[box2d]>=0.29.0
```

### Issue 4: GPU out of memory during training
**Solution**: Reduce batch size in config.json
```json
{
  "vae_trainer": {
    "batch_size": 50  // Reduce from 100
  },
  "mdrnn_trainer": {
    "batch_size": 10  // Reduce from 25
  }
}
```

### Issue 5: Data generation takes too long
**Solution**: Reduce rollouts in config.json
```json
{
  "data_generator": {
    "rollouts": 1000  // Reduce from 10000
  }
}
```

### Issue 6: TensorBoard shows no metrics
**Solution**: Ensure `is_logging: true` in config
```json
{
  "test_suite": {
    "is_logging": true
  }
}
```

### Issue 7: "is_slippery" parameter error in FrozenLake
**Solution**: This is fixed in test files; ensure you're using gymnasium imports
```python
import gymnasium as gym  # NOT: import gym
```

## 💾 Backing Up Models

Important: Always back up trained models before running long jobs.

```bash
# Backup MDRNN
cp mdrnn/checkpoints/MY_MODEL_mdrnn_best.tar mdrnn/checkpoints/backups/MY_MODEL_mdrnn_best.tar.bak

# Backup VAE
cp vae/checkpoints/MY_MODEL_vae_best.tar vae/checkpoints/backups/MY_MODEL_vae_best.tar.bak

# Backup iteration results
cp -r mdrnn/iteration_stats/ mdrnn/iteration_stats_backup_$(date +%s)/
```

## 🖥️ Running on Headless Servers

If running on a server without display:

```bash
# Install xvfb (virtual display)
sudo apt-get install xvfb

# Run with virtual display
xvfb-run -a -s "-screen 0 1280x1024x24" python main.py

# Or use provided script
bash run_headless.sh
```

## 📈 Performance Optimization

### GPU Acceleration
```bash
# Automatically uses GPU if available (CUDA 11.8+)
# Verify GPU is detected:
python -c "import torch; print(torch.cuda.is_available())"
```

### CPU Optimization
For multi-threaded trials:
```json
{
  "test_suite": {
    "is_multithread_trials": true,
    "fixed_cores": 8  // Use 8 cores
  }
}
```

### Memory Optimization
```json
{
  "vae_trainer": {
    "train_buffer_size": 25,  // Reduce from 50
    "test_buffer_size": 25
  }
}
```

## 📝 Complete Example Workflow

```bash
# 1. Install dependencies
conda create -n world_model python=3.9
conda activate world_model
pip install torch>=2.0 torchvision>=0.15
pip install gymnasium[box2d]>=0.29.0 numpy>=1.26.0 matplotlib tensorboard tqdm colorama

# 2. Generate data (optional - can use pre-generated)
# [Edit config.json: is_generate_data=true]
# python main.py
# Wait 1-4 hours...

# 3. Train VAE
# [Edit config.json: is_train_vae=true]
python main.py
# Wait 2-4 hours...

# 4. Train MDRNN
# [Edit config.json: is_train_vae=false, is_train_mdrnn=true]
python main.py
# Wait 3-6 hours...

# 5. Run planning tests
# [Edit config.json: is_train_mdrnn=false, is_run_planning_tests=true]
python main.py
# Wait 2-4 hours...

# 6. Monitor results
tensorboard --logdir utility/logging/tensorboard_runs --port 6006
# Open: http://localhost:6006
```

## 📚 Advanced Configuration

### Planning Agent Parameters

**RMHC (Random Mutation Hill Climbing)**:
```json
{
  "planning_agent": "RMHC",
  "random_mutation_hill_climb": {
    "horizon": 20,           // Look-ahead steps
    "max_generations": 12,   // Mutation generations
    "is_shift_buffer": true, // Reuse previous actions
    "is_rollout": false,
    "max_rollouts": 1,
    "rollout_length": 20
  }
}
```

**RHEA (Rolling Horizon Evolutionary Algorithm)**:
```json
{
  "planning_agent": "RHEA",
  "rolling_horizon": {
    "population_size": 4,    // Population size
    "horizon": 20,           // Look-ahead steps
    "max_generations": 5,    // Evolution generations
    "is_shift_buffer": true,
    "is_rollout": false,
    "max_rollouts": 1
  }
}
```

**MCTS (Monte Carlo Tree Search)**:
```json
{
  "planning_agent": "MCTS",
  "monte_carlo_tree_search": {
    "max_rollouts": 100,     // Rollouts per action
    "rollout_length": 20,    // Simulation length
    "temperature": 1.41      // Exploration parameter
  }
}
```

## ✅ Checklist Before Running

- [ ] Python 3.9+ installed
- [ ] gymnasium >= 0.29.0 installed
- [ ] PyTorch >= 2.0 installed
- [ ] config.json properly configured
- [ ] Appropriate feature flags set (`is_train_vae`, `is_train_mdrnn`, etc.)
- [ ] Output directories writable
- [ ] Sufficient disk space (20GB+ for full pipeline)
- [ ] RAM >= 8GB (16GB recommended)
- [ ] For GPU: CUDA 11.8+ available

## 🎓 Key Project Concepts

**VAE (Variational Autoencoder)**: Learns compact latent representation of game frames
- Input: 64×64×3 RGB images
- Latent: 64-dimensional vector
- Reconstruction: Predicts next frame from latent state

**MDRNN (Mixture Density RNN)**: Learns world dynamics in latent space
- Input: Current latent state + action
- Output: Next latent state distribution (5 Gaussians) + reward + termination

**Planning Algorithms**: Search for action sequences in the learned world model
- **RMHC**: Fast hill-climbing mutations
- **RHEA**: Population-based evolution
- **MCTS**: Tree-based sampling

**Iterative Training**: Refines MDRNN using planning agent's collected trajectories

## 📞 Support

For detailed model architecture information, see:
- [VAE Architecture](vae/vae.py)
- [MDRNN Architecture](mdrnn/mdrnn.py)
- [Planning Algorithms](planning/simulation/)

For paper reference:
- Original Paper: https://arxiv.org/abs/2011.11293
- World Models: https://doi.org/10.5281/zenodo.1207631
- RHEA Algorithm: https://arxiv.org/pdf/2003.12331.pdf

---

**Last Updated**: May 2024  
**Compatible With**: Python 3.9+, PyTorch 2.x, Gymnasium 1.1.1+
