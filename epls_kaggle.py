#!/usr/bin/env python
# coding: utf-8
# ==============================================================================
# EPLS: Evolutionary Planning in Latent Space
# Complete Reproduction Notebook — Table 1 (Olesen et al., 2020)
# Targets: Random=356, Iterative=708, Expert=765
# ==============================================================================

# %% [markdown]
# ## Cell 1: Install Dependencies
# %%
import subprocess, sys, os, json, shutil, time, glob

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout: print(result.stdout[-2000:])
    if result.returncode != 0 and result.stderr:
        print("[STDERR]", result.stderr[-500:])
    return result.returncode

run("apt-get install -y xvfb swig > /dev/null 2>&1")
# Cài đặt swig bản python và ép cài box2d để tránh lỗi DependencyNotInstalled
run(f"{sys.executable} -m pip install -q swig")
run(f"{sys.executable} -m pip install -q 'gymnasium[box2d]'")
run(f"{sys.executable} -m pip install -q shimmy[gym-v21] dill colorama")
print("✅ Dependencies installed (Box2D fixed)")

# %% [markdown]
# ## Cell 2: Setup Project & Helpers
# %%
DATASET_PATH = "/kaggle/input/epls-world-model/WorldModelPlanning"
WORK_DIR     = "/kaggle/working/WorldModelPlanning"

if not os.path.exists(WORK_DIR):
    shutil.copytree(DATASET_PATH, WORK_DIR)
    print(f"✅ Project initialized at {WORK_DIR}")
else:
    print(f"🔄 Resuming session in {WORK_DIR}")

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

# ---------- Monkey Patch cho Gym/Gymnasium Compatibility ----------
# Cấu trúc Wrapper thủ công để đảm bảo tương thích mọi phiên bản (Gym/Gymnasium, Python 3.10/3.12)
try:
    import gymnasium as gym
    ENV_ID = "CarRacing-v2"
    print("✨ Using Gymnasium (CarRacing-v2)")
except ImportError:
    import gym
    ENV_ID = "CarRacing-v0"
    print("✨ Using Legacy Gym (CarRacing-v0)")

class StepCompatibilityWrapper(gym.Wrapper):
    """Ép kết quả trả về của step() luôn là 4 giá trị (Legacy API)"""
    def step(self, action):
        results = self.env.step(action)
        if len(results) == 5:
            # obs, reward, terminated, truncated, info
            return results[0], results[1], results[2] or results[3], results[4]
        return results

original_make = gym.make
def compatible_make(id, **kwargs):
    target_id = ENV_ID if id == "CarRacing-v0" else id
    env = original_make(target_id, **kwargs)
    return StepCompatibilityWrapper(env)

gym.make = compatible_make
import sys
sys.modules['gym'] = gym 
print("✅ Manual StepCompatibilityWrapper applied")

# ---------- Xvfb helper ----------
_xvfb_proc = None
def start_xvfb():
    global _xvfb_proc
    if _xvfb_proc is None:
        _xvfb_proc = subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x1024x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["DISPLAY"] = ":99"
        time.sleep(1)
        print("✅ Xvfb started")

# ---------- Config helpers ----------
def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base (modifies base in-place)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def set_config(template_file: str, overrides: dict = None) -> dict:
    """Copy template → config.json, apply deep overrides, return final cfg."""
    shutil.copy(template_file, "config.json")
    with open("config.json") as f:
        cfg = json.load(f)
    if overrides:
        _deep_merge(cfg, overrides)
        with open("config.json", "w") as f:
            json.dump(cfg, f, indent=4)
    return cfg

def copy_checkpoint(src_name: str, dst_name: str):
    """Copy model checkpoints (MDRNN + VAE) from src experiment to dst."""
    for model_dir, suffix in [("mdrnn", "mdrnn_best.tar"), ("vae", "vae_best.tar")]:
        src = f"{model_dir}/checkpoints/{src_name}_{suffix}"
        dst = f"{model_dir}/checkpoints/{dst_name}_{suffix}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"  📋 {src} → {dst}")

def checkpoint_exists(experiment_name: str, model="mdrnn") -> bool:
    path = f"{model}/checkpoints/{experiment_name}_{model}_best.tar"
    return os.path.exists(path)

start_xvfb()
print("✅ Helpers ready")

# ==============================================================================
# STEP 1 — Non-Iterative RANDOM Model  (Target: 356 ± 177)
# ==============================================================================
# %% [markdown]
# ## Step 1: Random Model — Data Generation, Training, Benchmark
# %%
print("\n" + "="*60)
print("STEP 1: Non-Iterative Random Model  (Target: ~356)")
print("="*60)

# ── 1A. Generate 5,000 random rollouts (Optimized for Kaggle 20GB limit) ──────
existing_random = len(glob.glob("data_random_raw/*.npz"))
if existing_random < 5000:
    print(f"⏳ Found {existing_random}/5000 rollouts. Generating rest...")
    set_config("config_kaggle_p1_generate.json", {
        "experiment_name": "World_Model_Random",
        "is_generate_data": True,
        "data_generator": {
            "data_output_dir": "data_random_raw",
            "rollouts": 5000,
            "sequence_length": 501,
            "car_racing": {"is_ha_agent_driver": False}
        }
    })
    exec(open("main.py").read())
else:
    print(f"✅ {existing_random} random rollouts found — skipping generation.")

# ── 1B. Train VAE  (latent_size=64, 20 epochs) ───────────────────────────────
if not checkpoint_exists("World_Model_Random", "vae"):
    print("🚀 Training VAE (Random)...")
    set_config("config_phase1_benchmark.json", {
        "experiment_name": "World_Model_Random",
        "is_generate_data": False,
        "is_train_vae": True,
        "is_train_mdrnn": False,
        "latent_size": 64,
        "vae_trainer": {"max_epochs": 20, "is_continue_model": False}
    })
    exec(open("main.py").read())
else:
    print("✅ VAE (Random) checkpoint found — skipping.")

# ── 1C. Train MDRNN  (hidden_units=256, 60 epochs, seq_len=500) ──────────────
if not checkpoint_exists("World_Model_Random", "mdrnn"):
    print("🚀 Training MDRNN (Random)...")
    set_config("config_phase1_benchmark.json", {
        "experiment_name": "World_Model_Random",
        "is_generate_data": False,
        "is_train_vae": False,
        "is_train_mdrnn": True,
        "mdrnn": {"hidden_units": 256, "num_gaussians": 5},
        "mdrnn_trainer": {
            "max_epochs": 60,
            "sequence_length": 500,
            "is_continue_model": False,
            "early_stop_after_n_bad_epochs": 5
        }
    })
    exec(open("main.py").read())
else:
    print("✅ MDRNN (Random) checkpoint found — skipping.")

# ── 1D. Benchmark  (RMHC, horizon=20, gen=10, 100 trials) ────────────────────
print("📊 Benchmarking Random Model (RMHC h=20, g=10, 100 trials)...")
set_config("config_phase1_benchmark.json", {
    "experiment_name": "World_Model_Random",
    "is_generate_data": False,
    "is_train_vae": False,
    "is_train_mdrnn": False,
    "test_suite": {
        "is_run_planning_tests": True,
        "trials": 100,
        "is_reload_planning_session": False
    },
    "planning": {
        "planning_agent": "RMHC",
        "random_mutation_hill_climb": {
            "horizon": 20,
            "max_generations": 10,
            "is_shift_buffer": True
        }
    }
})
exec(open("main.py").read())
print("✅ Step 1 done — check planning_test_results/ for scores.")

# ==============================================================================
# STEP 2 — Iterative Model A  (Target: 708 ± 195)
# ==============================================================================
# %% [markdown]
# ## Step 2: Iterative Model — 5 Iterations, Replay Buffer, Benchmark
# %%
print("\n" + "="*60)
print("STEP 2: Iterative Model A  (Target: ~708)")
print("="*60)

# ── 2A. Initialise Iterative checkpoints from Random Baseline ─────────────────
print("🚚 Copying Random → Iter_A checkpoints (if needed)...")
copy_checkpoint("World_Model_Random", "World_Model_Iter_A")

# ── 2B. Run 5 Iterative Training rounds ──────────────────────────────────────
# Resume-safe: IterativeTrainer reads iteration_stats to skip completed rounds.
set_config("config_kaggle_p3_iterative.json", {
    "experiment_name": "World_Model_Iter_A",
    "forced_vae": "World_Model_Iter_A",
    "is_generate_data": False,
    "is_train_vae": False,
    "is_train_mdrnn": False,
    "is_iterative_train_mdrnn": True,
    "mdrnn": {"hidden_units": 256, "num_gaussians": 5},
    "iterative_trainer": {
        "num_iterations": 5,
        "num_rollouts": 500,
        "sequence_length": 250,
        "max_epochs": 10,
        "replay_buffer": {
            "is_replay_buffer": True,
            "max_buffer_size": 50000
        }
    },
    # Planning used DURING data collection inside each iteration
    "planning": {
        "planning_agent": "RMHC",
        "random_mutation_hill_climb": {
            "horizon": 20,
            "max_generations": 15,
            "is_shift_buffer": True
        }
    }
})
exec(open("main.py").read())

# ── 2C. Final Benchmark with "golden params" (h=20, g=15, 100 trials) ────────
print("📊 Final Benchmark — Iterative Model (RMHC h=20, g=15, 100 trials)...")
set_config("config_phase3_final_benchmark.json", {
    "experiment_name": "iterative_World_Model_Iter_A",
    "forced_vae": "World_Model_Iter_A",
    "is_generate_data": False,
    "is_train_vae": False,
    "is_train_mdrnn": False,
    "is_iterative_train_mdrnn": False,
    "test_suite": {
        "is_run_planning_tests": True,
        "trials": 100,
        "is_reload_planning_session": False
    },
    "mdrnn": {"hidden_units": 256},
    "planning": {
        "planning_agent": "RMHC",
        "random_mutation_hill_climb": {
            "horizon": 20,
            "max_generations": 15,
            "is_shift_buffer": True
        }
    }
})
exec(open("main.py").read())
print("✅ Step 2 done.")

# ==============================================================================
# STEP 3 — Non-Iterative EXPERT Model  (Target: 765 ± 102)
# ==============================================================================
# %% [markdown]
# ## Step 3: Expert Model — Mixed Data (5k random + 5k expert), Train, Benchmark
# %%
print("\n" + "="*60)
print("STEP 3: Non-Iterative Expert Model  (Target: ~765)")
print("="*60)

# ── 3A. Generate 5,000 extra random rollouts (reuse existing if possible) ─────
existing_random = len(glob.glob("data_random_raw/*.npz"))
if existing_random < 5000:
    print(f"⏳ Need 5k random rollouts (have {existing_random}). Generating...")
    set_config("config_kaggle_p1_generate.json", {
        "experiment_name": "World_Model_Expert",
        "is_generate_data": True,
        "data_generator": {
            "data_output_dir": "data_random_raw",
            "rollouts": 5000 - existing_random,
            "sequence_length": 501,
            "car_racing": {"is_ha_agent_driver": False}
        }
    })
    exec(open("main.py").read())
else:
    print(f"✅ {min(existing_random, 5000)} random rollouts ready for Expert mix.")

# ── 3B. Generate 5,000 expert rollouts (Ha Agent) ────────────────────────────
existing_expert = len(glob.glob("data_expert_raw/*.npz"))
if existing_expert < 5000:
    print(f"⏳ Found {existing_expert}/5000 expert rollouts. Generating rest...")
    set_config("config_kaggle_p1_generate.json", {
        "experiment_name": "World_Model_Expert",
        "is_generate_data": True,
        "data_generator": {
            "data_output_dir": "data_expert_raw",
            "rollouts": 5000 - existing_expert,
            "sequence_length": 501,
            "car_racing": {"is_ha_agent_driver": True}
        }
    })
    exec(open("main.py").read())
else:
    print(f"✅ {min(existing_expert, 5000)} expert rollouts found — skipping.")

# ── 3C. Merge data: copy first 5k random + 5k expert into data_mixed/ ─────────
os.makedirs("data_mixed_expert", exist_ok=True)
mixed_files = glob.glob("data_mixed_expert/*.npz")
if len(mixed_files) < 10000:
    print("🔀 Building mixed dataset (5k random + 5k expert)...")
    # Take up to 5000 from each source
    random_files = sorted(glob.glob("data_random_raw/*.npz"))[:5000]
    expert_files = sorted(glob.glob("data_expert_raw/*.npz"))[:5000]
    for src in random_files + expert_files:
        dst = os.path.join("data_mixed_expert", os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    print(f"✅ Mixed dataset: {len(glob.glob('data_mixed_expert/*.npz'))} files")
else:
    print(f"✅ Mixed dataset ready ({len(mixed_files)} files).")

# ── 3D. Train VAE + MDRNN on mixed data ──────────────────────────────────────
if not checkpoint_exists("World_Model_Expert", "vae"):
    print("🚀 Training VAE (Expert)...")
    set_config("config_phase1_benchmark.json", {
        "experiment_name": "World_Model_Expert",
        "is_generate_data": False,
        "is_train_vae": True,
        "is_train_mdrnn": False,
        "latent_size": 64,
        "data_dir": "data_mixed_expert",
        "vae_trainer": {"max_epochs": 20, "is_continue_model": False}
    })
    exec(open("main.py").read())
else:
    print("✅ VAE (Expert) checkpoint found — skipping.")

if not checkpoint_exists("World_Model_Expert", "mdrnn"):
    print("🚀 Training MDRNN (Expert)...")
    set_config("config_phase1_benchmark.json", {
        "experiment_name": "World_Model_Expert",
        "is_generate_data": False,
        "is_train_vae": False,
        "is_train_mdrnn": True,
        "data_dir": "data_mixed_expert",
        "mdrnn": {"hidden_units": 256, "num_gaussians": 5},
        "mdrnn_trainer": {
            "max_epochs": 60,
            "sequence_length": 500,
            "is_continue_model": False,
            "early_stop_after_n_bad_epochs": 5
        }
    })
    exec(open("main.py").read())
else:
    print("✅ MDRNN (Expert) checkpoint found — skipping.")

# ── 3E. Benchmark Expert Model (RMHC h=20, g=15, 100 trials) ─────────────────
print("📊 Benchmarking Expert Model (RMHC h=20, g=15, 100 trials)...")
set_config("config_phase1_benchmark.json", {
    "experiment_name": "World_Model_Expert",
    "is_generate_data": False,
    "is_train_vae": False,
    "is_train_mdrnn": False,
    "test_suite": {
        "is_run_planning_tests": True,
        "trials": 100,
        "is_reload_planning_session": False
    },
    "mdrnn": {"hidden_units": 256},
    "planning": {
        "planning_agent": "RMHC",
        "random_mutation_hill_climb": {
            "horizon": 20,
            "max_generations": 15,
            "is_shift_buffer": True
        }
    }
})
exec(open("main.py").read())
print("✅ Step 3 done.")

# ==============================================================================
# RESULTS SUMMARY
# ==============================================================================
# %% [markdown]
# ## Final Summary
# All benchmark scores are written to: `planning_test_results/`
# Compare your output with Table 1 from Olesen et al. (2020):
#
# | Method                              | Paper Score  |
# |-------------------------------------|-------------|
# | DQN [16]                            | 343 ± 18    |
# | Non-Iterative Random Model  (Ours)  | 356 ± 177   |
# | A3C Continuous [9]                  | 591 ± 45    |
# | Iterative Model A (5 it.)   (Ours)  | 708 ± 195   |
# | Non-Iterative Expert Model  (Ours)  | 765 ± 102   |
# | World Model [6]                     | 906 ± 21    |
