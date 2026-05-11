# Script vs Notebook Comparison

Quick comparison of `epls_kaggle.py` (Python script) vs `epls_kaggle.ipynb` (Jupyter notebook).

## 📋 Feature Comparison

| Feature | Script (.py) | Notebook (.ipynb) |
|---------|--------------|-------------------|
| **Execution Method** | `!python epls_kaggle.py` | Run → Run All (or cell-by-cell) |
| **Language** | Pure Python | Jupyter/IPython |
| **Progress Tracking** | ✅ Yes | ✅ Yes |
| **Timeout Recovery** | ✅ Yes | ✅ Yes |
| **Customization** | Edit text file | Edit code cells |
| **Debugging** | Run, view output | Step through cells |
| **Monitoring** | Watch terminal output | Watch notebook output |
| **Documentation** | Comments in code | Markdown + code cells |
| **Interactivity** | None (linear execution) | High (cell-by-cell) |
| **Mobile-friendly** | No | Yes (Kaggle interface) |
| **Easy to share** | Yes (one file) | Yes (one file) |
| **IDE support** | VS Code, PyCharm | Kaggle, Jupyter, Colab |
| **Environment vars** | `EPLS_RUN_PHASE=1` | Set `RUN_PHASE = "1"` |

## 🎯 When to Use Each

### Use Script (epls_kaggle.py) When:
- ✅ Running in **terminal/command-line**
- ✅ **Automating** with cron jobs or CI/CD
- ✅ **Batch processing** multiple runs
- ✅ **Minimal overhead** needed
- ✅ Want **traditional execution flow**
- ✅ Using **non-Jupyter environment**

### Use Notebook (epls_kaggle.ipynb) When:
- ✅ Using **Kaggle directly** (recommended)
- ✅ Want **visual progress monitoring**
- ✅ Need to **debug interactively**
- ✅ Want **detailed documentation** inline
- ✅ Prefer **cell-by-cell execution**
- ✅ Need **easy customization** interface
- ✅ Want to **add analysis cells** alongside training

## 🚀 Usage Comparison

### Running Phase 1 with Script
```bash
export EPLS_RUN_PHASE=1
python epls_kaggle.py
```

### Running Phase 1 with Notebook
```python
# Cell 1 (at top): Set
RUN_PHASE = "1"

# Then: Run All cells
# Or: Click "Run All" button
```

### Running Phase 2 with Script
```bash
export EPLS_RUN_PHASE=2
python epls_kaggle.py
```

### Running Phase 2 with Notebook
```python
# Cell 1 (at top): Set
RUN_PHASE = "2"

# Then: Run All cells
```

## 📁 File Structure

### Script Version
```
WorldModelPlanning/
├── epls_kaggle.py          (510 lines)
├── main.py
├── config.json
└── [other project files]
```

### Notebook Version
```
WorldModelPlanning/
├── epls_kaggle.ipynb       (JSON format, 27 cells)
├── main.py
├── config.json
└── [other project files]
```

## 💾 Size Comparison

| Metric | Script | Notebook |
|--------|--------|----------|
| File Size | ~20 KB | ~45 KB |
| Lines of Code | 510 | 470 (in cells) |
| Markdown Documentation | 0 | 12 cells |
| Easy to Read | Yes | Very Yes |

## 🔄 Progress Tracking

Both use **identical progress tracking**:
```json
{
  "phase": 1,
  "stage": "1B_vae_trained",
  "completed_tasks": ["phase1_generate_data", "phase1_train_vae"],
  "timestamp": "2024-05-11 03:45:00"
}
```

## ⏱️ Execution Time

**No difference** - both execute identical code.

- Phase 1: 8-10 hours
- Phase 2: 8-12 hours
- Total: 16-22 hours (across multiple Kaggle sessions)

## 🛠️ Customization

### Modifying Batch Size

**Script:**
```python
# Edit epls_kaggle.py, find line ~284:
"batch_size": 100,  # Change to 50
```

**Notebook:**
```python
# Edit Cell 7, find line:
"batch_size": 100,  # Change to 50
# Then re-run Cell 7
```

### Changing Number of Iterations

**Script:**
```python
# Edit epls_kaggle.py, find line ~399:
"num_iterations": 5,  # Change to 10
```

**Notebook:**
```python
# Edit Cell 11, find line:
"num_iterations": 5,  # Change to 10
# Then re-run Cell 11
```

## 🐛 Debugging

### Script Debugging
```bash
# Add print statements in epls_kaggle.py
# Re-run: python epls_kaggle.py
# See output in terminal
```

### Notebook Debugging
```python
# Add code cell to inspect:
import json
with open("config.json") as f:
    cfg = json.load(f)
print(cfg["mdrnn_trainer"])

# Run cell with Shift+Enter
# See output inline
```

## 📊 Monitoring

### Script
```
Terminal output streams continuously
Watch for:
- ✅ Progress saved messages
- ⏳ Task timing
- ❌ Errors printed to stderr
```

### Notebook
```
Cell output appears below each cell
Easy to see:
- ✅ Which tasks completed
- 📊 Real-time training metrics
- ❌ Error messages with full context
- 📈 Progress visualization
```

## 🔗 Integration

### Script with Other Tools
```bash
# Easy to pipe/redirect output
python epls_kaggle.py > training.log 2>&1

# Easy to schedule
crontab -e
# 0 9 * * * cd /kaggle/working && python epls_kaggle.py

# Easy to integrate with CI/CD
# .github/workflows/training.yml
# ... script: python epls_kaggle.py
```

### Notebook with Kaggle
```
Upload to Kaggle directly
No additional setup needed
Kaggle handles execution scheduling
Auto-saves progress
```

## ✨ Advantages of Each

### Script Advantages
- Simpler environment setup
- Works in any terminal
- Easy to automate
- Minimal file size
- Traditional workflow
- Works on any machine (local, HPC cluster, etc.)

### Notebook Advantages
- Native Kaggle integration
- Beautiful inline documentation
- Interactive debugging
- Easy to add analysis cells
- Built-in progress visualization
- Can stop/resume mid-cell
- Natural for exploration
- Better for sharing with teammates

## 🎓 Learning & Documentation

### Script
- Organized with comments
- Linear execution flow
- Self-contained file
- Good for CI/CD learning

### Notebook
- Markdown cells explain each section
- Visual execution with outputs
- Cell structure organizes logic
- Better for learning how it works
- Easier to follow along

## 🔄 Converting Between Them

### Python Script → Notebook
Already done! See `epls_kaggle.ipynb`
- 13 cells organized by function
- Markdown titles for each cell
- Identical functionality

### Notebook → Python Script
Could be done with:
```bash
jupyter nbconvert --to script epls_kaggle.ipynb
# Creates: epls_kaggle.py
```

## 📊 Performance

**Identical performance** - both run same Python code

```
CPU Usage: ~same
GPU Usage: ~same
Memory Usage: ~same
Execution Time: ~same
```

## 🎯 Recommendation

| Use Case | Recommendation |
|----------|---|
| **Kaggle Notebook** | Use `.ipynb` (notebook) |
| **Local Machine** | Use `.py` (script) or `.ipynb` |
| **HPC Cluster** | Use `.py` (script) |
| **Learning** | Use `.ipynb` (notebook) |
| **Production** | Use `.py` (script) |
| **Team Sharing** | Use `.ipynb` (notebook) |
| **Automation** | Use `.py` (script) |

## 📚 Both Files Included

You get **both**:
- `epls_kaggle.py` - Python script version
- `epls_kaggle.ipynb` - Jupyter notebook version

Choose whichever fits your workflow!

---

**Summary**: Same functionality, different interfaces. Pick based on where you're running it.
