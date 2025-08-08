PyTorch implementation of the paper "[Adaptive k-space Radial Sampling for Cardiac MRI with Reinforcement Learning](http://arxiv.org/abs/2508.04727)".

## 📁 Project Structure

```
cardiac-mri-rl/
├── src/
│   ├── models/
│   │   ├── dual_branch_network.py    # Main network architecture
│   │   ├── cross_attention.py        # Cross-attention fusion module
│   │   └── transformer_encoder.py    # Transformer-based encoder
│   ├── sampling/
│   │   ├── golden_angle.py          # Golden angle sampling strategy
│   │   └── mask_generation.py       # K-space mask generation
│   ├── training/
│   │   ├── ppo_trainer.py           # PPO training implementation
│   │   ├── reward_functions.py      # Anatomically-aware rewards
│   │   └── environment.py           # MRI reconstruction environment
│   └── utils/
│       ├── data_loader.py           # ACDC dataset handling
│       ├── metrics.py               # Evaluation metrics (SSIM, PSNR, Dice)
│       └── visualization.py         # Result visualization tools
├── configs/
│   ├── training_config.yaml         # Training hyperparameters
│   └── model_config.yaml           # Model architecture settings
├── experiments/
│   ├── ablation_study.py           # Ablation experiments
│   └── comparison_baselines.py     # Baseline comparisons
├── notebooks/
│   ├── golden_angle_analysis.ipynb # Mathematical analysis
│   └── results_visualization.ipynb # Results analysis
├── data/
│   └── acdc/                       # ACDC dataset (download separately)
├── outputs/
│   ├── models/                     # Trained model checkpoints
│   ├── logs/                       # Training logs
│   └── figures/                    # Generated visualizations
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### 1. Data Preparation

Download the ACDC dataset and place it in the `data/acdc/` directory:

```bash
# Download ACDC dataset from official source
# https://www.creatis.insa-lyon.fr/Challenge/acdc/
```

### 2. Training

```bash
# Train the model with default settings
python src/training/ppo_trainer.py --config configs/training_config.yaml

# Train with custom acceleration factor
python src/training/ppo_trainer.py --acceleration 12 --batch_size 30
```

### 3. Evaluation

```bash
# Evaluate trained model
python src/evaluation/evaluate.py --model_path outputs/models/best_model.pth --test_data data/acdc/test/

# Generate comparison figures
python src/utils/visualization.py --results_dir outputs/results/
```

### 4. Golden Angle Analysis

```bash
# Generate golden angle mathematical analysis
python experiments/golden_angle_analysis.py
```
