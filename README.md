# Enhancing Agent Performance in VizDoom via Deep Reinforcement Learning

This repository contains the implementation of autonomous agents trained to navigate and combat in first-person 3D environments using **Deep Reinforcement Learning**.

### 🚀 Research Overview
- **Environment:** ViZDoom (ZDoom-based research platform).
- **Algorithms:** Comparison of PPO (On-policy) vs. DDDQN (Off-policy). A2C included as a baseline.
- **Key Result:** Achieved a stable 50 FPS throughput and a 12% increase in reward-based decision accuracy.

### 📂 Repository Structure
- `/src`: Core Python source code and modules (consolidation point for algorithms).
- `/notebooks`: Jupyter notebooks for experiments and analysis.
- `/config`: Configuration files (e.g., VizDoom scenarios, hyperparameters).
- `/results`: TensorBoard logs, CSV/JSON metrics, and generated plots.

> Existing algorithm folders (A2C, DDDQN, PPO) are preserved for traceability; future commits may consolidate them under `/src`.

### 🧠 Model Weights
Due to the high complexity of the trained states, the final weight files (9.2 GB) are hosted externally.
- [Link to External Storage/Drive/HuggingFace]

### 📊 Performance
The agent demonstrates superior navigation in the **Deadly Corridor** scenario, utilizing PPO for stable gradient updates in high-dimensional state spaces.

### 🛠️ Setup
```bash
pip install -r requirements.txt
```

### ⚠️ Note on Large Files
This repository uses a strict `.gitignore` policy to exclude large/binary artifacts (model checkpoints, zips, caches). This keeps the repo lightweight and reviewable.
