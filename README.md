# 🛰️ Land Cover Segmentation – GUI Application

This project implements **semantic segmentation** on the [LoveDA Dataset](https://github.com/Junjue-Wang/LoveDA) using a **U-Net with ResNet50 encoder**.  
It provides a **Tkinter-based GUI** for easy visualization of segmentation results, making it suitable for **satellite image analysis** in:
- Urban–Rural Planning
- Agricultural Monitoring
- Road Infrastructure Mapping

---

## 📌 Key Features

### 7-Class Land Cover Segmentation:
1. **Background**
2. **Building**
3. **Forest**
4. **Road**
5. **Water**
6. **Barren**
7. **Agricultural**

- **Distinct Color Mapping** for easy interpretation  
- **Tkinter GUI** for side-by-side comparison:
  - Original satellite image
  - Segmentation output
  - Color-coded class legend
- **Supports Pretrained Model** – `loveda_resnet50_256_finetuned.pth`

---

## 🛠️ Built With
- **PyTorch**
- **Segmentation Models PyTorch (SMP)**
- **Albumentations**
- **OpenCV**
- **Pillow (PIL)**

---

## 📂 Project Structure
```
├── train.py                  # Training script
├── test.py                   # Testing script with visualization
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/
│   ├── train/                # Training images and masks
│   ├── val/                  # Validation images and masks
│   └── test/                 # Test images and masks
│
├── outputs/
│   ├── predictions/          # Predicted masks
│   └── results.png            # Sample output visualization
│
└── saved_models/
    └── best_model.pth         # Trained model weights
```
---
##📌 **Requirements**
- **Python 3.8+**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Matplotlib**
---
## **Output Result**
![Urban Segmentation Result](https://example.com/image.png)

---
