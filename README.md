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
## **Left: Original Image | Right: Segmentation Output | Bottom: Legend**
- **Sample Result 1**
 <img src="Screenshot 2025-08-15 123244.png" alt="Segmentation Output" width="750">
 
- **Sample Result 2**
 <img src="Screenshot 2025-08-15 130319.png" alt="Segmentation Output" width="750">


## 🎯**Application Relevance**
**This system aligns with Remote Sensing mission areas:**
- Remote Sensing & GIS Analysis
- Land Use and Land Cover (LULC) Mapping
- Urban-Rural Infrastructure Planning
- Agricultural Resource Management
- Water Body and Road Network Monitoring
## **Advantages:**
- High accuracy from deep learning models
- Simple GUI for field operators
- Customizable for different satellite datasets
## 📜 License
MIT License – free to use and adapt for research and development.

## 👨‍💻 Author

Eshank Ryshabh
📧 ryshabheshank@gmail.com
🔗 GitHub Profile [Profile Link](https://github.com/Junjue-Wang/LoveDA)

