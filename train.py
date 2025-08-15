import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ---------------- CONFIG ----------------
NUM_CLASSES = 7
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 150
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\LoveDA\train\images"
TRAIN_MASK_DIR = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\LoveDA\train\masks"
VAL_IMG_DIR = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\LoveDA\val\images"
VAL_MASK_DIR = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\LoveDA\val\masks"

MODEL_SAVE_PATH = "loveda_resnet50_256_finetuned.pth"

# ---------------- DATASET ----------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png").replace(".jpeg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

# ---------------- TRANSFORMS ----------------
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ---------------- DATA LOADERS ----------------
train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- MODEL ----------------
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",  # Use pretrained encoder
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

# Loss & optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ---------------- TRAINING LOOP ----------------
best_val_loss = float("inf")
patience_counter = 0
EARLY_STOPPING_PATIENCE = 7

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Saved Best Model at epoch {epoch+1}")
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("⏹ Early stopping triggered.")
        break

print("Training complete.")
