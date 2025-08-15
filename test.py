import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ---------------- CONFIG ----------------
NUM_CLASSES = 7
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "loveda_resnet50_256_finetuned.pth"  # trained model path

CLASS_NAMES = ['Background', 'Building', 'Forest', 'Road', 'Water', 'Barren', 'Agricultural']
COLORS = np.array([
    [0, 0, 0],         # Background - Black
    [255, 255, 0],     # Building - Yellow
    [128, 64, 128],    # Road - Purple
    [0, 0, 255],       # Water - Blue
    [210, 180, 140],   # Barren - Tan
    [34, 139, 34],     # Forest - Green
    [255, 165, 0]      # Agricultural - Orange
], dtype=np.uint8)

# ---------------- LOAD MODEL ----------------
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- TRANSFORM ----------------
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    sample = transform(image=resized_img)
    img_tensor = sample['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Colorize prediction
    color_mask = COLORS[pred_mask]
    blended = cv2.addWeighted(resized_img, 0.6, color_mask.astype(np.uint8), 0.4, 0)

    return resized_img, blended

# ---------------- LEGEND GENERATOR ----------------
def create_legend():
    legend_height = 25 * len(CLASS_NAMES)
    legend_width = 200
    legend_img = Image.new("RGB", (legend_width, legend_height), (255, 255, 255))
    draw = ImageDraw.Draw(legend_img)

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        y = i * 25
        draw.rectangle([5, y + 5, 25, y + 20], fill=tuple(color.tolist()))
        draw.text((35, y + 5), cls_name, fill=(0, 0, 0))

    return legend_img

# ---------------- GUI ----------------
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    original, prediction = predict_image(file_path)

    # Convert to Tkinter Image
    orig_img = Image.fromarray(original)
    pred_img = Image.fromarray(prediction)
    legend_img = create_legend()

    orig_img_tk = ImageTk.PhotoImage(orig_img)
    pred_img_tk = ImageTk.PhotoImage(pred_img)
    legend_img_tk = ImageTk.PhotoImage(legend_img)

    # Show in GUI
    original_label.config(image=orig_img_tk)
    original_label.image = orig_img_tk

    prediction_label.config(image=pred_img_tk)
    prediction_label.image = pred_img_tk

    legend_label.config(image=legend_img_tk)
    legend_label.image = legend_img_tk

# Tkinter window
root = tk.Tk()
root.title("LoveDA Segmentation Prediction")

# Upload button
btn = tk.Button(root, text="Upload Image", command=upload_and_predict, font=("Arial", 12), bg="#4CAF50", fg="white")
btn.pack(pady=10)

# Layout frames
frame_images = tk.Frame(root)
frame_images.pack()

original_label = tk.Label(frame_images)
original_label.pack(side="left", padx=10, pady=10)

prediction_label = tk.Label(frame_images)
prediction_label.pack(side="left", padx=10, pady=10)

legend_label = tk.Label(root)
legend_label.pack(pady=10)

root.mainloop()

