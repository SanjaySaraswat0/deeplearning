# MNIST Classification using MLP (Deep Learning Case Study 1)

## 📌 Objective
The objective of this case study is to implement a **Multi-Layer Perceptron (MLP)**
for handwritten digit classification using the **MNIST dataset**, and to analyze
model performance by applying different **variations and hyperparameters**.

---

## 📊 Dataset
- **Dataset:** MNIST Handwritten Digits
- **Image Size:** 28 × 28 (grayscale)
- **Classes:** Digits 0–9
- **Total Samples:** 70,000  
  - Training: 60,000  
  - Testing: 10,000  

The dataset is loaded using TensorFlow Keras built-in utilities.

---

## ⚙️ Preprocessing Steps
- Reshaped images from `28×28` to `784`
- Normalized pixel values to range `[0, 1]`
- Used train-test split provided by MNIST

---

## 🧠 Model Architecture (MLP)
- Input Layer: 784 neurons  
- Hidden Layer 1: 128 neurons  
- Hidden Layer 2: 64 neurons  
- Output Layer: 10 neurons  
- Activation Functions: ReLU / Sigmoid / Tanh (based on experiment)
- Output Activation: Softmax  

---

## 🔁 Variations Implemented

### 1️⃣ Activation Function Variations
- `relu.py`
- `sigmoid.py`
- `tanh.py`

**Purpose:**  
To analyze the effect of different activation functions on convergence speed and accuracy.

---

### 2️⃣ Optimizer Variations
- `optimizer_adam.py`
- `optimizer_sgd.py`

**Purpose:**  
To compare optimization strategies and their impact on training efficiency.

---

### 3️⃣ Epoch Variations
- `epochs_5.py`
- `epochs_10.py`
- `epochs_20.py`

**Purpose:**  
To observe underfitting, optimal fitting, and overfitting behavior.

---

## 📁 Project Structure
