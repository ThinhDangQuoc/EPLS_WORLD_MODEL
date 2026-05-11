#!/usr/bin/env python
# coding: utf-8
# ==============================================================================
# EPLS: Evolutionary Planning in Latent Space - Kaggle Edition
# Paper: Olesen et al., 2020 - https://arxiv.org/abs/2011.11293
# Modernized for: Python 3.9+, PyTorch 2.x, Gymnasium 1.1.1+
# ==============================================================================

import subprocess, sys, os, json, shutil, time, glob

# ==============================================================================
# KAGGLE EXECUTION CONTROL
# Set RUN_PHASE to "1", "2", or "all" to run specific workflow stages
RUN_PHASE = os.environ.get("EPLS_RUN_PHASE", "all")
# ==============================================================================

def run(cmd):
    """Execute shell command and stream output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0 and result.stderr:
        print("[STDERR]", result.stderr[-500:])
    return result.returncode

# Install system dependencies and Python packages
print("📦 Installing dependencies...")
run("apt-get install -y xvfb swig > /dev/null 2>&1")
run(f"{sys.executable} -m pip install -q swig")
run(f"{sys.executable} -m pip install -q 'gymnasium[box2d]>=0.29.0'")
run(f"{sys.executable} -m pip install -q dill colorama")
print("✅ Dependencies installed (Gymnasium 1.1.1+ with Box2D)")

def run_main():
    """Run main.py with streaming output for real-time monitoring."""
    process = subprocess.Popen(f"{sys.executable} main.py", shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
    returncode = process.wait()
    if returncode != 0:
        print(f"\n❌ Error: main.py exited with code {returncode}")
    return returncode

# ============================================================================
# PROJECT SETUP
# ============================================================================
GIT_REPO = "https://github.com/ThinhDangQuoc/EPLS_WORLD_MODEL.git"
WORK_DIR = "/kaggle/working/WorldModelPlanning"

# Clone or update repository
if not os.path.exists(WORK_DIR):
    print(f"🚀 Cloning repository from {GIT_REPO}...")
    run(f"git clone {GIT_REPO} {WORK_DIR}")
else:
    print(f"🔄 Updating repository...")
    os.chdir(WORK_DIR)
    run("git checkout -- .")
    run("git pull")

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

# Link previous Kaggle phase outputs
def link_previous_kaggle_outputs():
    input_dir = "/kaggle/input"
    if not os.path.exists(input_dir):
        return
    print("🔗 Linking previous Kaggle outputs...")
    folders_to_link = [
        "data_random_raw", "data_expert_raw", "data_mixed_expert",
        "data_iterative", "mdrnn/checkpoints", "vae/checkpoints"
    ]
    linked_files = 0
    for root, dirs, files in os.walk(input_dir):
        if root.count(os.sep) - input_dir.count(os.sep) > 3:
            continue
        for target in folders_to_link:
            folder_name = os.path.basename(target)
            if folder_name in dirs:
                src_path = os.path.join(root, folder_name)
                dst_path = os.path.join(WORK_DIR, target)
                os.makedirs(dst_path, exist_ok=True)
                for file_name in os.listdir(src_path):
                    if not file_name.endswith(('.npz', '.tar', '.json')):
                        continue
                    src_file = os.path.join(src_path, file_name)
                    dst_file = os.path.join(dst_path, file_name)
                    if os.path.isfile(src_file) and not os.path.exists(dst_file):
                        os.symlink(src_file, dst_file)
                        linked_files += 1
    if linked_files > 0:
        print(f"  ✅ Linked {linked_files} files from previous runs")

# Ensure all directories are Python packages
def fix_python_packages():
    print("🛠️ Verifying Python packages...")
    created = 0
    for root, dirs, files in os.walk(WORK_DIR):
        if ".git" in root or "__pycache__" in root:
            continue
        if "__init__.py" not in files:
            with open(os.path.join(root, "__init__.py"), "w") as f:
                pass
            created += 1
    if created:
        print(f"  🆕 Created {created} missing __init__.py files")
    for mod in list(sys.modules.keys()):
        if any(pkg in mod for pkg in ["mdrnn", "vae", "utility", "planning", "environment"]):
            del sys.modules[mod]
    print("  🧹 Module cache cleared")

link_previous_kaggle_outputs()
fix_python_packages()
print(f"✨ Using Gymnasium (CarRacing-v3) with native 5-tuple step API")

# ---------- Xvfb helper ----------
_xvfb_proc = None
def start_xvfb():
    """Start Xvfb virtual display for headless Kaggle environment."""
    global _xvfb_proc
    if _xvfb_proc is None:
        try:
            _xvfb_proc = subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1280x1024x24"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.environ["DISPLAY"] = ":99"
            time.sleep(2)
            if _xvfb_proc.poll() is not None:
                print("⚠️ Xvfb failed to start — continuing headless")
                _xvfb_proc = None
            else:
                print("✅ Xvfb virtual display started")
        except FileNotFoundError:
            print("⚠️ Xvfb not available — continuing headless")

# ---------- Config helpers ----------
def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base config."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def set_config(overrides: dict) -> dict:
    """Load config.json, apply overrides, and save."""
    with open("config.json") as f:
        cfg = json.load(f)
    if overrides:
        _deep_merge(cfg, overrides)
        with open("config.json", "w") as f:
            json.dump(cfg, f, indent=4)
    return cfg

def copy_checkpoint(src_name: str, dst_name: str):
    """Copy model checkpoints (VAE + MDRNN) from src to dst experiment."""
    for model_dir, suffix in [("vae", "vae_best.tar"), ("mdrnn", "mdrnn_best.tar")]:
        src = f"{model_dir}/checkpoints/{src_name}_{suffix}"
        dst = f"{model_dir}/checkpoints/{dst_name}_{suffix}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"  📋 Copied {src_name} → {dst_name}")

def checkpoint_exists(experiment_name: str, model: str = "mdrnn") -> bool:
    """Check if experiment checkpoint exists."""
    path = f"{model}/checkpoints/{experiment_name}_{model}_best.tar"
    return os.path.exists(path)

def ensure_dirs():
    """Create all required directories."""
    dirs = [
        "vae/checkpoints", "mdrnn/checkpoints", "mdrnn/checkpoints/backups",
        "data_random_raw", "data_expert_raw", "data_mixed_expert", "data_iterative",
        "tests_custom/planning_test_results", "mdrnn/iteration_stats", "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directory structure ready")

# ---------- Progress tracking ----------
PROGRESS_FILE = "kaggle_progress.json"

def load_progress():
    """Load progress state from previous run."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "phase": 1,
        "stage": None,
        "completed_tasks": [],
        "timestamp": None
    }

def save_progress(phase: int, stage: str, completed: list):
    """Save current progress state."""
    progress = {
        "phase": phase,
        "stage": stage,
        "completed_tasks": completed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"  📝 Progress saved: Phase {phase}, Stage: {stage}")

def is_task_done(task_name: str, progress: dict) -> bool:
    """Check if task was already completed."""
    return task_name in progress.get("completed_tasks", [])

def mark_task_done(task_name: str, progress: dict) -> dict:
    """Mark task as completed in progress."""
    if task_name not in progress["completed_tasks"]:
        progress["completed_tasks"].append(task_name)
    return progress

start_xvfb()
ensure_dirs()
progress = load_progress()
print(f"✅ Helpers ready")
print(f"📊 Progress: Phase {progress['phase']}, Last stage: {progress.get('stage', 'None')}")
print(f"   Completed tasks: {len(progress['completed_tasks'])}\n")

# ==============================================================================
# WORKFLOW PHASES
# ==============================================================================

if RUN_PHASE in ['all', '1']:
    print("="*70)
    print("PHASE 1: Random Baseline Model (Target: ~356 avg reward)")
    print("="*70)

    # Load progress and skip completed tasks
    if progress["phase"] > 1:
        print("⏭️  Phase 1 already completed in previous run")
    elif progress["phase"] == 1:
        print(f"📋 Resuming Phase 1 from previous run")
        print(f"   Completed: {', '.join(progress['completed_tasks']) or 'None'}\n")

        # 1A. Generate random rollouts
        if not is_task_done("phase1_generate_data", progress):
            print("⏳ Task 1A: Generating random rollouts...")
            existing_random = len(glob.glob("data_random_raw/*.npz"))
            if existing_random < 5000:
                print(f"   Found {existing_random}/5000. Generating {5000-existing_random} more...")
                set_config({
                    "experiment_name": "World_Model_Random",
                    "is_generate_data": True,
                    "is_train_vae": False,
                    "is_train_mdrnn": False,
                    "data_generator": {
                        "data_output_dir": "data_random_raw",
                        "data_prefix": "random",
                        "rollouts": 5000,
                        "sequence_length": 501,
                        "car_racing": {"is_ha_agent_driver": False}
                    }
                })
                if run_main() != 0:
                    print("🛑 Task 1A failed. Stopping execution to prevent invalid progress saving.")
                    sys.exit(1)
            else:
                print(f"   ✅ {existing_random} random rollouts found\n")
            progress = mark_task_done("phase1_generate_data", progress)
            save_progress(1, "1A_data_generated", progress["completed_tasks"])
        else:
            print("✅ Task 1A: Data generation (already done)\n")

        # 1B. Train VAE
        if not is_task_done("phase1_train_vae", progress):
            print("🚀 Task 1B: Training VAE (Random, 20 epochs)...")
            if checkpoint_exists("World_Model_Random", "vae"):
                print("   ℹ️  VAE checkpoint exists, skipping training\n")
            else:
                set_config({
                    "experiment_name": "World_Model_Random",
                    "is_generate_data": False,
                    "is_train_vae": True,
                    "is_train_mdrnn": False,
                    "is_iterative_train_mdrnn": False,
                    "latent_size": 64,
                    "vae_trainer": {
                        "max_epochs": 20,
                        "batch_size": 100,
                        "learning_rate": 0.0001,
                        "is_continue_model": False
                    }
                })
                if run_main() != 0:
                    print("🛑 Task 1B failed. Stopping execution to prevent invalid progress saving.")
                    sys.exit(1)
            progress = mark_task_done("phase1_train_vae", progress)
            save_progress(1, "1B_vae_trained", progress["completed_tasks"])
        else:
            print("✅ Task 1B: VAE training (already done)\n")

        # 1C. Train MDRNN
        if not is_task_done("phase1_train_mdrnn", progress):
            print("🚀 Task 1C: Training MDRNN (Random, 60 epochs)...")
            if checkpoint_exists("World_Model_Random", "mdrnn"):
                print("   ℹ️  MDRNN checkpoint exists, skipping training\n")
            else:
                set_config({
                    "experiment_name": "World_Model_Random",
                    "is_generate_data": False,
                    "is_train_vae": False,
                    "is_train_mdrnn": True,
                    "is_iterative_train_mdrnn": False,
                    "latent_size": 64,
                    "mdrnn": {"hidden_units": 256, "num_gaussians": 5},
                    "mdrnn_trainer": {
                        "max_epochs": 60,
                        "batch_size": 25,
                        "learning_rate": 0.001,
                        "sequence_length": 500,
                        "is_continue_model": False,
                        "early_stop_after_n_bad_epochs": 5
                    }
                })
                if run_main() != 0:
                    print("🛑 Task 1C failed. Stopping execution to prevent invalid progress saving.")
                    sys.exit(1)
            progress = mark_task_done("phase1_train_mdrnn", progress)
            save_progress(1, "1C_mdrnn_trained", progress["completed_tasks"])
        else:
            print("✅ Task 1C: MDRNN training (already done)\n")

        # 1D. Benchmark with RMHC
        if not is_task_done("phase1_benchmark", progress):
            print("📊 Task 1D: Benchmarking Random Model (RMHC, 100 trials)...")
            set_config({
                "experiment_name": "World_Model_Random",
                "is_generate_data": False,
                "is_train_vae": False,
                "is_train_mdrnn": False,
                "is_iterative_train_mdrnn": False,
                "latent_size": 64,
                "planning": {
                    "planning_agent": "RMHC",
                    "random_mutation_hill_climb": {
                        "horizon": 20,
                        "max_generations": 10,
                        "is_shift_buffer": True,
                        "is_rollout": False,
                        "max_rollouts": 1
                    }
                },
                "test_suite": {
                    "is_run_planning_tests": True,
                    "is_run_model_tests": False,
                    "trials": 100,
                    "is_multithread_trials": False,
                    "is_logging": True
                }
            })
            if run_main() != 0:
                print("🛑 Task 1D failed. Stopping execution to prevent invalid progress saving.")
                sys.exit(1)
            progress = mark_task_done("phase1_benchmark", progress)
            save_progress(2, "phase1_complete", progress["completed_tasks"])
        else:
            print("✅ Task 1D: Benchmarking (already done)\n")

        print("="*70)
        print("✅ PHASE 1 COMPLETE")
        print("="*70)
        print("Random Model results in: tests_custom/planning_test_results/\n")

# ==============================================================================
if RUN_PHASE in ['all', '2']:
    print("="*70)
    print("PHASE 2: Iterative Refinement (Target: ~708 avg reward)")
    print("="*70)

    # Load/create progress for Phase 2
    if progress["phase"] < 2:
        print("⏳ Phase 1 not completed yet. Run Phase 1 first or set EPLS_RUN_PHASE=all\n")
    else:
        print(f"📋 Phase {progress['phase']} status")
        print(f"   Last stage: {progress.get('stage', 'None')}")
        print(f"   Completed tasks: {len(progress['completed_tasks'])}\n")

        # 2A. Copy Random baseline → Iter_A baseline
        if not is_task_done("phase2_setup_checkpoint", progress):
            print("🚚 Task 2A: Setting up iterative model from Random baseline...")
            copy_checkpoint("World_Model_Random", "World_Model_Iter_A")
            progress = mark_task_done("phase2_setup_checkpoint", progress)
            save_progress(2, "2A_checkpoint_setup", progress["completed_tasks"])
        else:
            print("✅ Task 2A: Checkpoint setup (already done)\n")

        # 2B. Run iterative training
        if not is_task_done("phase2_iterative_train", progress):
            print("🔄 Task 2B: Running iterative training (5 iterations)...")
            set_config({
                "experiment_name": "World_Model_Iter_A",
                "forced_vae": "World_Model_Iter_A",
                "is_generate_data": False,
                "is_train_vae": False,
                "is_train_mdrnn": False,
                "is_iterative_train_mdrnn": True,
                "latent_size": 64,
                "mdrnn": {"hidden_units": 256, "num_gaussians": 5},
                "iterative_trainer": {
                    "iterative_data_dir": "data_iterative",
                    "num_iterations": 5,
                    "num_rollouts": 500,
                    "sequence_length": 250,
                    "max_epochs": 10,
                    "test_scenario": "planning_whole_random_track",
                    "replay_buffer": {
                        "is_replay_buffer": True,
                        "max_buffer_size": 50000
                    }
                },
                "planning": {
                    "planning_agent": "RMHC",
                    "random_mutation_hill_climb": {
                        "horizon": 20,
                        "max_generations": 15,
                        "is_shift_buffer": True,
                        "is_rollout": False,
                        "max_rollouts": 1
                    }
                }
            })
            if run_main() != 0:
                print("🛑 Task 2B failed. Stopping execution to prevent invalid progress saving.")
                sys.exit(1)
            progress = mark_task_done("phase2_iterative_train", progress)
            save_progress(2, "2B_iterative_trained", progress["completed_tasks"])
        else:
            print("✅ Task 2B: Iterative training (already done)\n")

        # 2C. Benchmark iterative model
        if not is_task_done("phase2_benchmark", progress):
            print("📊 Task 2C: Benchmarking Iterative Model (RMHC, 100 trials)...")
            set_config({
                "experiment_name": "World_Model_Iter_A",
                "is_generate_data": False,
                "is_train_vae": False,
                "is_train_mdrnn": False,
                "is_iterative_train_mdrnn": False,
                "latent_size": 64,
                "planning": {
                    "planning_agent": "RMHC",
                    "random_mutation_hill_climb": {
                        "horizon": 20,
                        "max_generations": 10,
                        "is_shift_buffer": True
                    }
                },
                "test_suite": {
                    "is_run_planning_tests": True,
                    "is_run_model_tests": False,
                    "trials": 100,
                    "is_logging": True
                }
            })
            if run_main() != 0:
                print("🛑 Task 2C failed. Stopping execution to prevent invalid progress saving.")
                sys.exit(1)
            progress = mark_task_done("phase2_benchmark", progress)
            save_progress(3, "phase2_complete", progress["completed_tasks"])
        else:
            print("✅ Task 2C: Benchmarking (already done)\n")

        print("="*70)
        print("✅ PHASE 2 COMPLETE")
        print("="*70)
        print("Iterative Model results in: tests_custom/planning_test_results/\n")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("✨ KAGGLE EXECUTION SUMMARY")
print("="*70)

# Load final progress
final_progress = load_progress()
print(f"\n📊 Progress Report:")
print(f"   Current Phase: {final_progress['phase']}")
print(f"   Last Stage: {final_progress.get('stage', 'None')}")
print(f"   Completed Tasks ({len(final_progress['completed_tasks'])}):")
for task in final_progress["completed_tasks"]:
    print(f"     ✓ {task}")

expected_tasks = {
    1: ["phase1_generate_data", "phase1_train_vae", "phase1_train_mdrnn", "phase1_benchmark"],
    2: ["phase2_setup_checkpoint", "phase2_iterative_train", "phase2_benchmark"]
}
current_expected = expected_tasks.get(final_progress["phase"], [])
remaining = [t for t in current_expected if t not in final_progress["completed_tasks"]]

if remaining:
    print(f"\n⏳ Remaining tasks ({len(remaining)}):")
    for task in remaining:
        print(f"     ⋯ {task}")
else:
    print(f"\n✅ All Phase {final_progress['phase']} tasks completed!")

print(f"\n📁 Results Locations:")
print(f"   - Model checkpoints: mdrnn/checkpoints/")
print(f"   - Benchmarks: tests_custom/planning_test_results/")
print(f"   - Iteration stats: mdrnn/iteration_stats/")
print(f"   - Progress tracker: {PROGRESS_FILE}")

print(f"\n🎯 Paper Targets:")
print(f"   - Random Model: ~356 ± 177 avg reward")
print(f"   - Iterative Model: ~708 ± 195 avg reward")
print(f"   - Expert Model: ~765 (optional)")

print(f"\n⚙️  Resumption Instructions:")
print(f"   If Kaggle times out:")
print(f"   1. Run again: EPLS_RUN_PHASE=all python epls_kaggle.py")
print(f"   2. Script automatically resumes from last completed task")
print(f"   3. Check {PROGRESS_FILE} for detailed progress")

print(f"\n📚 For detailed instructions, see: EXECUTION_GUIDE.md")
print("="*70)
