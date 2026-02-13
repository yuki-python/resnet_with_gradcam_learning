
import os
import sys
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datetime import datetime


# ==========================
# 設定
# ==========================
USE_GRADCAM_REG = False
NUM_EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4
IMAGE_SIZE = (224, 224)
LOG_FILE = "training_log.txt"
CHECKPOINT_DIR = "checkpoints"
TRAIN_ROOT = "dataset/train"
SPLIT_RATIO = 0.7

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Tee class with timestamp
# ------------------------------------------------------------
class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, "a", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        if data.strip() != "":
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.stdout.write(timestamp + data)
            self.file.write(timestamp + data)
        else:
            self.stdout.write(data)
            self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


# ------------------------------------------------------------
# Load Labelme folder
# ------------------------------------------------------------
def load_labelme_folder(folder):
    samples = []
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return samples

    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    print(f"[INFO] Loading Labelme folder: {folder}")
    print(f"[INFO] Found {len(files)} JSON files")

    for i, file in enumerate(files):
        json_path = os.path.join(folder, file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = os.path.join(folder, data["imagePath"])
        label = 1 if len(data.get("shapes", [])) > 0 else 0

        polygons = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "polygon":
                polygons.append(shape["points"])

        samples.append({
            "image_path": image_path,
            "label": label,
            "positive_polygons": polygons
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  Loaded {i+1}/{len(files)} files")

    return samples


def load_all_train_samples(train_root):
    pos_dir = os.path.join(train_root, "pocket_positive_labelme")
    neg_dir = os.path.join(train_root, "pocket_negative_labelme")

    samples = []
    samples += load_labelme_folder(pos_dir)
    samples += load_labelme_folder(neg_dir)

    print(f"[INFO] Total train samples (before split): {len(samples)}")
    return samples


def split_train_val(samples, ratio=0.7, seed=42):
    random.Random(seed).shuffle(samples)
    n_total = len(samples)
    n_train = int(n_total * ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    print(f"[INFO] Split train/val = {len(train_samples)} / {len(val_samples)}")
    return train_samples, val_samples


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class PolygonCamDataset(Dataset):
    def __init__(self, samples, transform=None, image_size=(224,224), use_masks=False):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size
        self.use_masks = use_masks
        print(f"[INFO] Dataset initialized with {len(samples)} samples (use_masks={use_masks})")

    def __len__(self):
        return len(self.samples)

    def _make_masks(self, img_w, img_h, polygons, label):
        if label == 0:
            pos_mask = np.zeros((img_h, img_w), dtype=np.float32)
            neg_mask = np.ones((img_h, img_w), dtype=np.float32)
            return pos_mask, neg_mask

        pos_mask_img = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(pos_mask_img)

        for poly in polygons:
            draw.polygon(poly, outline=1, fill=1)

        pos_mask = np.array(pos_mask_img, dtype=np.float32)
        neg_mask = 1.0 - pos_mask
        return pos_mask, neg_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        w, h = img.size

        img = img.resize(self.image_size)

        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        label = torch.tensor(sample["label"], dtype=torch.long)

        if not self.use_masks:
            return img_t, label

        fixed_polygons = [
            [(float(x), float(y)) for x, y in poly]
            for poly in sample["positive_polygons"]
        ]

        pos_mask, neg_mask = self._make_masks(w, h, fixed_polygons, sample["label"])

        pos_mask = Image.fromarray((pos_mask * 255).astype(np.uint8)).resize(self.image_size)
        neg_mask = Image.fromarray((neg_mask * 255).astype(np.uint8)).resize(self.image_size)

        pos_mask = torch.from_numpy(np.array(pos_mask) / 255.0).unsqueeze(0)
        neg_mask = torch.from_numpy(np.array(neg_mask) / 255.0).unsqueeze(0)

        return img_t, label, pos_mask, neg_mask


# ------------------------------------------------------------
# ResNet + Grad-CAM
# ------------------------------------------------------------
class ResNetWithGradCAM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # ★ 修正：最後の conv 層に hook を付ける
        self._features = None
        target_layer = self.model.layer4[-1].conv3
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, x):
        self._features = None
        return self.model(x)

    def compute_gradcam(self, class_scores, input_size):
        grads = torch.autograd.grad(
            outputs=class_scores.sum(),
            inputs=self._features,
            create_graph=True
        )[0]

        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * self._features).sum(dim=1)
        cam = F.relu(cam)

        cam = F.interpolate(cam.unsqueeze(1), size=input_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        B = cam.size(0)
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        cam_norm = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
        return cam_norm.view_as(cam)


# ------------------------------------------------------------
# CAM losses
# ------------------------------------------------------------
def cam_region_losses(cam, pos_mask, neg_mask):
    cam = cam.unsqueeze(1)

    pos_sum = (pos_mask * cam).sum(dim=(1,2,3))
    pos_area = pos_mask.sum(dim=(1,2,3)) + 1e-6
    pos_loss = - (pos_sum / pos_area).mean()

    neg_sum = (neg_mask * cam).sum(dim=(1,2,3))
    neg_area = neg_mask.sum(dim=(1,2,3)) + 1e-6
    neg_loss = (neg_sum / neg_area).mean()

    return pos_loss, neg_loss


# ------------------------------------------------------------
# Utility: accuracy
# ------------------------------------------------------------
def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / (total + 1e-8)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_model():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")
    print(f"[INFO] USE_GRADCAM_REG = {USE_GRADCAM_REG}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    all_samples = load_all_train_samples(TRAIN_ROOT)
    train_samples, val_samples = split_train_val(all_samples, ratio=SPLIT_RATIO)

    train_dataset = PolygonCamDataset(
        train_samples,
        transform=transform,
        image_size=IMAGE_SIZE,
        use_masks=USE_GRADCAM_REG
    )
    val_dataset = PolygonCamDataset(
        val_samples,
        transform=transform,
        image_size=IMAGE_SIZE,
        use_masks=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = ResNetWithGradCAM(num_classes=2).to(device)

    # ★ 上位層だけファインチューニングする設定
    for name, param in model.model.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    mode_tag = "with_gradcam" if USE_GRADCAM_REG else "plain"

    for epoch in range(NUM_EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{NUM_EPOCHS} ==========")
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            if USE_GRADCAM_REG:
                imgs, labels, pos_masks, neg_masks = batch
                pos_masks, neg_masks = pos_masks.to(device), neg_masks.to(device)
            else:
                imgs, labels = batch

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(imgs)
            loss_cls = criterion(logits, labels)

            if USE_GRADCAM_REG:
                class_scores = logits.gather(1, labels.view(-1,1)).squeeze(1)
                cam = model.compute_gradcam(class_scores, IMAGE_SIZE)
                loss_pos, loss_neg = cam_region_losses(cam, pos_masks, neg_masks)
                loss = loss_cls + loss_pos + loss_neg
            else:
                loss = loss_cls

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += compute_accuracy(logits, labels)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  [Train] Batch {batch_idx+1}/{total_batches} | Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / total_batches
        epoch_train_acc = running_acc / total_batches
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        val_batches = len(val_loader)

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)

                val_running_loss += loss.item()
                val_running_acc += compute_accuracy(logits, labels)

        epoch_val_loss = val_running_loss / val_batches
        epoch_val_acc = val_running_acc / val_batches
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"[INFO] Epoch {epoch+1} finished.")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val   Loss: {epoch_val_loss:.4f} | Val   Acc: {epoch_val_acc:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"model_{mode_tag}_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_path = os.path.join(
                CHECKPOINT_DIR,
                f"best_model_{mode_tag}.pth"
            )
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Updated best model: {best_path} (Val Acc: {best_val_acc:.4f})")

    # Plot curves
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve ({mode_tag})")
    plt.savefig(f"loss_curve_{mode_tag}.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy Curve ({mode_tag})")
    plt.savefig(f"accuracy_curve_{mode_tag}.png", bbox_inches="tight")
    plt.close()

    # Total time
    elapsed = time.time() - start_time
    print(f"[INFO] Training finished. Best Val Acc: {best_val_acc:.4f}")
    print(f"[INFO] Total training time: {elapsed:.2f} seconds")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    Tee(LOG_FILE)
    train_model()
