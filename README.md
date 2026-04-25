# 🧠 Self-Pruning Neural Network (CIFAR-10)

A PyTorch implementation of a **self-pruning neural network** that learns to remove unnecessary weights during training using learnable gates and L1 regularization.

---

## 📌 Overview

Modern neural networks are often over-parameterized, making them inefficient for deployment. This project implements a **self-pruning mechanism** where the model dynamically learns which weights are important and eliminates the rest during training.

Unlike traditional pruning (post-training), this approach integrates pruning **directly into the learning process**.

---

## 🚀 Key Features

* ✅ Custom **PrunableLinear layer** (no use of `nn.Linear`)
* ✅ Learnable **gate parameters for each weight**
* ✅ **Sigmoid-based gating** (continuous → binary via STE)
* ✅ **L1 sparsity regularization**
* ✅ **Warmup training phase**
* ✅ **Gradient clipping + cosine scheduler**
* ✅ **Automatic pruning during training**
* ✅ Full visualization:

  * Accuracy curves
  * Sparsity curves
  * Gate distribution
  * λ trade-off analysis
* ✅ Auto-generated **Markdown report**

---

## 🏗️ Architecture

```
Input (32×32×3)
   ↓
Flatten (3072)
   ↓
PrunableLinear (3072 → 1024)
   ↓
BatchNorm + ReLU + Dropout
   ↓
PrunableLinear (1024 → 512)
   ↓
BatchNorm + ReLU + Dropout
   ↓
PrunableLinear (512 → 256)
   ↓
BatchNorm + ReLU
   ↓
PrunableLinear (256 → 10)
   ↓
Output (10 classes)
```

---

## ⚙️ How It Works

Each weight has a **learnable gate**:

```
gate_prob = sigmoid(temperature × gate_score)
```

Final weight used:

```
pruned_weight = weight × gate
```

### 🔥 Sparsity Loss

```
Total Loss = CrossEntropy + λ × L1(gate_probs)
```

* L1 encourages gates → 0
* Gates near 0 → weights removed

---

## 🧪 Experiment Setup

| Parameter  | Value            |
| ---------- | ---------------- |
| Dataset    | CIFAR-10         |
| Model      | MLP              |
| Epochs     | 20               |
| Batch Size | 128              |
| Optimizer  | Adam             |
| Scheduler  | CosineAnnealing  |
| λ values   | 1e-3, 1e-2, 1e-1 |

---

## 📊 Results

| λ    | Accuracy | Sparsity |
| ---- | -------- | -------- |
| 1e-3 | 56.11%   | 83.7%    |
| 1e-2 | 56.75%   | 93.7%    |
| 1e-1 | 55.23%   | 98.2%    |

👉 

---

## 📈 Observations

* Increasing λ → higher sparsity
* Slight drop in accuracy
* Strong trade-off between efficiency and performance

---

## 📊 Visualizations

Generated automatically:

* `accuracy_curves.png`
* `sparsity_curves.png`
* `gate_distribution.png`
* `lambda_tradeoff.png`
* `dashboard.png`

---

## 📂 Project Structure

```
├── self_pruning_network.py
├── outputs/
│   ├── accuracy_curves.png
│   ├── sparsity_curves.png
│   ├── gate_distribution.png
│   ├── lambda_tradeoff.png
│   ├── dashboard.png
│   └── report.md
└── README.md
```

---

## ▶️ How to Run

```bash
pip install torch torchvision matplotlib numpy

python self_pruning_network.py
```

---

## 📌 Case Study Alignment

This implementation fully satisfies the requirements:

* ✔ Custom prunable layer
* ✔ L1 sparsity loss
* ✔ CIFAR-10 training
* ✔ λ trade-off analysis
* ✔ Sparsity metric
* ✔ Visualization + report

👉 

---

## 🧠 Key Insights

* L1 regularization pushes gates toward zero
* STE enables training with binary pruning
* Most weights are redundant (~98% removable)
* Medium λ gives best trade-off

---

## 🚀 Future Improvements

* 🔹 CNN-based pruning (higher accuracy)
* 🔹 Structured pruning (channel/filter level)
* 🔹 Model export for deployment
* 🔹 Quantization + pruning

---

## 👨‍💻 Author

**Balaji K**
Software Engineering Student


