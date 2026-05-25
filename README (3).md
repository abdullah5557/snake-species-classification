# 🐍 Snake Species Classification using EfficientNetB3

A deep learning project that classifies **24 species of snakes** using Transfer Learning with EfficientNetB3. Achieves **~92% test accuracy** with a two-phase fine-tuning strategy.

---

## 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) based on **EfficientNetB3** (pretrained on ImageNet) to identify snake species from images. The model was trained and evaluated on a custom augmented dataset of 24 snake classes.

| Property         | Detail                        |
|------------------|-------------------------------|
| Model            | EfficientNetB3 (Transfer Learning) |
| Dataset          | 24 Snake Species              |
| Input Size       | 224 × 224 × 3                 |
| Train/Val/Test   | 70% / 15% / 15%               |
| Test Accuracy    | ~92%                          |
| Framework        | TensorFlow / Keras            |
| Platform         | Google Colab (GPU)            |

---

## 🗂️ Project Structure

```
snake-species-classification/
│
├── snake_projec_code_full.ipynb   # Main Jupyter Notebook (full pipeline)
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Files to ignore
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB3 (pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.4)
    ↓
Dense(24, Softmax)   ← 24 snake classes
```

---

## 🔁 Training Strategy

Training was done in **two phases**:

**Phase 1 — Warm-up (8 epochs)**
- Last 20 layers of EfficientNetB3 unfrozen
- Learning rate: `1e-4`

**Phase 2 — Fine-tuning (10 epochs)**
- Last 30 layers unfrozen
- Learning rate: `1e-5` (lower to prevent catastrophic forgetting)

**Callbacks used:**
- `EarlyStopping` (patience=5)
- `ReduceLROnPlateau` (factor=0.3, patience=3)
- `ModelCheckpoint` (saves best model)

---

## 📊 Data Augmentation

To handle class imbalance and increase diversity, aggressive augmentation was applied:

| Technique         | Value              |
|-------------------|--------------------|
| Rotation          | ±60°               |
| Width/Height Shift| 30%                |
| Zoom              | 0.7× to 1.4×       |
| Horizontal Flip   | ✅                 |
| Vertical Flip     | ✅                 |
| Brightness Range  | 0.5 to 1.5         |
| Channel Shift     | ±30                |
| Fill Mode         | Reflect            |

Each class was augmented to a **minimum of 100 images**.

---

## 📈 Results

- **Test Accuracy: ~92%**
- **Test Loss:** Low (refer to training curves in notebook)

### Per-Class Performance
Most classes achieved **≥ 90% accuracy** (green). A few reached 70–89% (orange). No class fell below 70%.

> Full confusion matrix and classification report are available inside the notebook.

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/snake-species-classification.git
cd snake-species-classification
```

### 2. Open in Google Colab
Upload `snake_projec_code_full.ipynb` to [Google Colab](https://colab.research.google.com/) and run cells in order.

### 3. Prepare Dataset
- Upload your snake dataset as a ZIP file to Google Drive
- Update the dataset path in the notebook:
```python
extract_to = "/content/Snakes_Dataset/Snakes"
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

See `requirements.txt` for full list. Main dependencies:

- Python 3.8+
- TensorFlow ≥ 2.10
- scikit-learn
- matplotlib
- seaborn
- numpy

---

## 🐍 Snake Classes (24 Species)

The model classifies 24 different snake species. Class names are derived from folder names in the dataset directory.

---

## 👤 Author

**Abdullah**  
BS Artificial Intelligence — University of Management and Technology (UMT)  
📍 Pakistan

---

## 📄 License

This project is for academic/educational purposes.

---

## 🙏 Acknowledgements

- [EfficientNet Paper — Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- TensorFlow / Keras team
- Google Colab for free GPU access
