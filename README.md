# Intro to Deep Learning & Backpropagation
### Data Science Retreat — Berlin

This course covers the foundations of deep learning, from a single artificial neuron to the Transformer architecture. Every concept is built step-by-step in code, starting from NumPy and progressing to PyTorch.

---

## Course Structure

| # | Notebook | Topics |
|---|---|---|
| 01 | Why Deep Learning? | Limits of traditional ML, feature learning, XOR problem |
| 02 | Artificial Neuron & Architecture | Single neuron, MLP, layers, depth vs width |
| 03 | Activation Functions | Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, vanishing gradients preview |
| 04 | Loss Functions | MSE, Binary Cross-Entropy, Categorical Cross-Entropy |
| 05 | Forward Propagation | Layer-by-layer pass, caching, batch computation |
| 05b | PyTorch Fundamentals & Tensors | Tensors, autograd, `nn.Module`, optimizers, SGD vs Adam |
| 06 | The Chain Rule | Derivatives, chain rule, computational graphs |
| 07 | Backpropagation Algorithm | Full from-scratch backprop, gradient checking |
| 08 | Gradient Descent & Optimizers | Batch/SGD/Mini-batch, Momentum, Adam, learning rate |
| 09 | Practical Training Challenges | Vanishing gradients, weight init, BatchNorm, Dropout |
| 10 | Putting It All Together | Full MNIST MLP in PyTorch, training loops, reading loss curves |
| 11 | Convolutional Neural Networks | Convolution, filters, pooling, CNN architecture, MLP vs CNN |
| 12 | Tensors & Embeddings | Discrete inputs, `nn.Embedding`, sentiment classification, PCA visualization |
| 13 | Fundamentals of Transformers | Self-attention, Q/K/V, multi-head attention, positional encoding, encoder block |

**Prerequisites:** Python basics, NumPy familiarity, traditional ML fundamentals (logistic regression, gradient descent, overfitting/underfitting).

---

## ⚠️ Important: PyTorch Installation

PyTorch is **not available on the default PyPI index** — it has its own package server. Installing it requires passing an `--index-url` flag. The right URL depends on your operating system and whether you have an NVIDIA GPU.

| Platform | Install command |
|---|---|
| Linux / Windows — CPU only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Linux / Windows — CUDA 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| Linux / Windows — CUDA 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| macOS (Apple Silicon or Intel) | `pip install torch torchvision` *(standard PyPI works on macOS)* |

Not sure which CUDA version you have? Run `nvidia-smi` in a terminal — the top-right corner shows the CUDA version.

---

## Installation

### Option 1 — pip

```bash
# 1. Clone or download this repository
git clone https://github.com/your-org/dsr-deep-learning.git
cd dsr-deep-learning

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (Command Prompt)
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

# 4. Install non-PyTorch dependencies
pip install -r requirements.txt

# 5. Install PyTorch — pick the right command for your platform (see table above)
#    CPU only (safe default for everyone):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. Launch Jupyter
jupyter notebook
```

---

### Option 2 — uv (recommended — much faster)

[uv](https://github.com/astral-sh/uv) is a modern Python package manager written in Rust. It resolves and installs packages 10–100× faster than pip and handles virtual environments automatically.

```bash
# 1. Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# winget install astral-sh.uv                      # Windows

# Restart your terminal after installing uv, then:

# 2. Clone or download this repository
git clone https://github.com/your-org/dsr-deep-learning.git
cd dsr-deep-learning

# 3. Create a virtual environment with a specific Python version
uv venv .venv --python 3.11

# 4. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 5. Install non-PyTorch dependencies
uv pip install -r requirements.txt

# 6. Install PyTorch — pick the right command for your platform (see table above)
#    CPU only (safe default for everyone):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 7. Launch Jupyter
jupyter notebook
```

---

## Verifying Your Installation

After installation, open a terminal (with the virtual environment active) and run:

```bash
python -c "
import torch, torchvision, numpy, matplotlib, sklearn
print(f'PyTorch:     {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'NumPy:       {numpy.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
print('All good! ✓')
"
```

You should see version numbers printed without errors. GPU will show `False` on CPU-only installs — that is fine for this course.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── 01_Why_Deep_Learning.ipynb
├── 02_Artificial_Neuron_and_Architecture.ipynb
├── 03_Activation_Functions.ipynb
├── 04_Loss_Functions.ipynb
├── 05_Forward_Propagation.ipynb
├── 05b_PyTorch_Fundamentals_and_Tensors.ipynb
├── 06_The_Chain_Rule.ipynb
├── 07_Backpropagation_Algorithm.ipynb
├── 08_Gradient_Descent_and_Optimizers.ipynb
├── 09_Practical_Training_Challenges.ipynb
├── 10_Putting_It_All_Together.ipynb
├── 11_Convolutional_Neural_Networks.ipynb
├── 12_Tensors_and_Embeddings.ipynb
└── 13_Fundamentals_of_Transformers.ipynb
```

---

## What You'll Build

By the end of the course you will have implemented from scratch:

- A **multilayer perceptron** in NumPy, including forward and backward passes
- A **backpropagation** engine with gradient checking
- **SGD, SGD+Momentum, and Adam** optimizers
- A **CNN** that classifies MNIST digits
- A **word embedding** model that learns semantic structure
- A **Transformer encoder** block with self-attention, positional encoding, and residual connections

---

## License

Course materials © Samson Afolabi. For educational use only.
