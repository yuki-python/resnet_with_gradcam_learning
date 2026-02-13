
# import os
# import json
# import argparse
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# import torch
# import torch.nn.functional as F
# from torchvision import transforms

# # ★ 学習コードと同じ ResNetWithGradCAM を import
# from train_model import ResNetWithGradCAM


# # ------------------------------------------------------------
# # Load Labelme folder
# # ------------------------------------------------------------
# def load_labelme_folder(folder):
#     samples = []
#     if not os.path.isdir(folder):
#         print(f"[WARN] Folder not found: {folder}")
#         return samples

#     files = [f for f in os.listdir(folder) if f.endswith(".json")]

#     print(f"[INFO] Loading Labelme folder: {folder} ({len(files)} jsons)")

#     for file in files:
#         json_path = os.path.join(folder, file)
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         image_path = os.path.join(folder, data["imagePath"])
#         label = 1 if len(data.get("shapes", [])) > 0 else 0

#         samples.append({
#             "image_path": image_path,
#             "label": label
#         })

#     return samples


# # ------------------------------------------------------------
# # Load validation samples
# # ------------------------------------------------------------
# def load_val_samples(root_val_dir):
#     pos_dir = os.path.join(root_val_dir, "pocket_positive_labelme")
#     neg_dir = os.path.join(root_val_dir, "pocket_negative_labelme")

#     samples = []
#     samples += load_labelme_folder(pos_dir)
#     samples += load_labelme_folder(neg_dir)

#     print(f"[INFO] Total evaluation samples: {len(samples)}")
#     return samples


# # ------------------------------------------------------------
# # Grad-CAM visualization
# # ------------------------------------------------------------
# def save_gradcam_image(img_pil, cam, out_path):
#     img = np.array(img_pil.resize(cam.shape[::-1])) / 255.0
#     heatmap = cam / (cam.max() + 1e-8)
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = plt.cm.jet(heatmap)[:, :, :3]

#     overlay = 0.5 * img + 0.5 * heatmap
#     overlay = np.clip(overlay, 0, 1)

#     plt.figure(figsize=(6, 6))
#     plt.imshow(overlay)
#     plt.axis("off")
#     plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
#     plt.close()


# # ------------------------------------------------------------
# # Evaluate model
# # ------------------------------------------------------------
# def evaluate_model(val_root, model_path, mode_tag="eval"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Evaluating on device: {device}")
#     print(f"[INFO] Loading model from: {model_path}")

#     # Load model
#     model = ResNetWithGradCAM(num_classes=2).to(device)
#     state = torch.load(model_path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # Preprocessing
#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406],
#                              std=[0.229,0.224,0.225]),
#     ])

#     # Load val data
#     samples = load_val_samples(val_root)

#     TP = FP = TN = FN = 0
#     y_true = []
#     y_score = []

#     # Output folder for Grad-CAM
#     cam_dir = os.path.join(val_root, f"gradcam_output_{mode_tag}")
#     os.makedirs(cam_dir, exist_ok=True)

#     for idx, sample in enumerate(samples):
#         img_pil = Image.open(sample["image_path"]).convert("RGB")
#         x = transform(img_pil).unsqueeze(0).to(device)

#         # -----------------------------
#         # 推論（no_grad）
#         # -----------------------------
#         with torch.no_grad():
#             logits = model(x)
#             probs = F.softmax(logits, dim=1)
#             pred = logits.argmax(dim=1).item()

#         gt = sample["label"]

#         # Save prediction score for ROC
#         y_true.append(gt)
#         y_score.append(float(probs[0][1]))

#         # Confusion matrix
#         if gt == 1 and pred == 1:
#             TP += 1
#         elif gt == 0 and pred == 1:
#             FP += 1
#         elif gt == 0 and pred == 0:
#             TN += 1
#         elif gt == 1 and pred == 0:
#             FN += 1

#         # -----------------------------
#         # Grad-CAM（勾配が必要 → 再 forward）
#         # -----------------------------
#         x2 = transform(img_pil).unsqueeze(0).to(device)
#         logits_cam = model(x2)

#         # ★ pred のスコアに対する Grad-CAM（あなたの目的通り）
#         class_scores = logits_cam[0, pred].unsqueeze(0)

#         cam = model.compute_gradcam(class_scores, (224,224))
#         cam_np = cam[0].detach().cpu().numpy()

#         # Save Grad-CAM image
#         base = os.path.splitext(os.path.basename(sample["image_path"]))[0]
#         out_cam_path = os.path.join(cam_dir, f"{base}_cam.jpg")
#         save_gradcam_image(img_pil, cam_np, out_cam_path)

#         if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
#             print(f"  Processed {idx+1}/{len(samples)} images")

#     # ------------------------------------------------------------
#     # Metrics
#     # ------------------------------------------------------------
#     precision = TP / (TP + FP + 1e-8)
#     recall = TP / (TP + FN + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)
#     accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

#     print("=== Evaluation Results ===")
#     print(f"Total samples: {len(samples)}")
#     print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"F1 Score:  {f1:.4f}")
#     print(f"Accuracy:  {accuracy:.4f}")

#     # ------------------------------------------------------------
#     # ROC curve
#     # ------------------------------------------------------------
#     from sklearn.metrics import roc_curve, auc

#     fpr, tpr, _ = roc_curve(y_true, y_score)
#     roc_auc = auc(fpr, tpr)

#     plt.figure(figsize=(6,6))
#     plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
#     plt.plot([0,1], [0,1], color="gray", lw=1, linestyle="--")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"ROC Curve ({mode_tag})")
#     plt.legend(loc="lower right")

#     roc_path = os.path.join(val_root, f"roc_curve_{mode_tag}.png")
#     plt.savefig(roc_path, bbox_inches="tight")
#     plt.close()

#     print(f"ROC curve saved to: {roc_path}")


# # ------------------------------------------------------------
# # Main
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True,
#                         help="Path to model checkpoint (.pth)")
#     parser.add_argument("--val_root", type=str, default="dataset/val",
#                         help="Path to validation dataset root")
#     parser.add_argument("--tag", type=str, default="eval",
#                         help="Tag name for output folders")
#     args = parser.parse_args()

#     evaluate_model(
#         val_root=args.val_root,
#         model_path=args.model,
#         mode_tag=args.tag
#     )





import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

from train_model import ResNetWithGradCAM


def load_labelme_folder(folder):
    samples = []
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return samples

    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    print(f"[INFO] Loading Labelme folder: {folder} ({len(files)} jsons)")

    for file in files:
        json_path = os.path.join(folder, file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = os.path.join(folder, data["imagePath"])
        label = 1 if len(data.get("shapes", [])) > 0 else 0

        samples.append({
            "image_path": image_path,
            "label": label
        })

    return samples


def load_val_samples(root_val_dir):
    pos_dir = os.path.join(root_val_dir, "pocket_positive_labelme")
    neg_dir = os.path.join(root_val_dir, "pocket_negative_labelme")

    samples = []
    samples += load_labelme_folder(pos_dir)
    samples += load_labelme_folder(neg_dir)

    print(f"[INFO] Total evaluation samples: {len(samples)}")
    return samples


def save_gradcam_image(img_pil, cam, out_path):
    img = np.array(img_pil.resize(cam.shape[::-1])) / 255.0
    heatmap = cam / (cam.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]

    overlay = 0.5 * img + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def evaluate_model(val_root, model_path, mode_tag="eval"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")
    print(f"[INFO] Loading model from: {model_path}")

    model = ResNetWithGradCAM(num_classes=2).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    samples = load_val_samples(val_root)

    TP = FP = TN = FN = 0
    y_true = []
    y_score = []

    # 出力フォルダ
    cam_root = os.path.join(val_root, f"gradcam_output_{mode_tag}")
    false_root = os.path.join(val_root, f"false_{mode_tag}")
    os.makedirs(cam_root, exist_ok=True)
    os.makedirs(false_root, exist_ok=True)

    categories = ["TP", "FP", "TN", "FN"]
    cam_dirs = {cat: os.path.join(cam_root, cat) for cat in categories}
    false_dirs = {cat: os.path.join(false_root, cat) for cat in ["FP", "FN"]}

    for d in cam_dirs.values():
        os.makedirs(d, exist_ok=True)
    for d in false_dirs.values():
        os.makedirs(d, exist_ok=True)

    counters = {cat: 0 for cat in categories}

    for idx, sample in enumerate(samples):
        img_pil = Image.open(sample["image_path"]).convert("RGB")
        x = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

        gt = sample["label"]
        y_true.append(gt)
        y_score.append(float(probs[0][1]))

        if gt == 1 and pred == 1:
            result = "TP"
            TP += 1
        elif gt == 0 and pred == 1:
            result = "FP"
            FP += 1
        elif gt == 0 and pred == 0:
            result = "TN"
            TN += 1
        elif gt == 1 and pred == 0:
            result = "FN"
            FN += 1

        # Grad-CAM
        x2 = transform(img_pil).unsqueeze(0).to(device)
        logits_cam = model(x2)
        class_scores = logits_cam[0, pred].unsqueeze(0)
        cam = model.compute_gradcam(class_scores, (224,224))
        cam_np = cam[0].detach().cpu().numpy()

        count = counters[result]
        filename = f"{result}_{mode_tag}_{count:03d}.jpg"
        counters[result] += 1

        cam_path = os.path.join(cam_dirs[result], filename)
        save_gradcam_image(img_pil, cam_np, cam_path)

        if result in ["FP", "FN"]:
            false_img_path = os.path.join(false_dirs[result], filename)
            save_gradcam_image(img_pil, cam_np, false_img_path)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            print(f"  Processed {idx+1}/{len(samples)} images")

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    print("=== Evaluation Results ===")
    print(f"Total samples: {len(samples)}")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({mode_tag})")
    plt.legend(loc="lower right")

    roc_path = os.path.join(val_root, f"roc_curve_{mode_tag}.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to: {roc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--val_root", type=str, default="dataset/val",
                        help="Path to validation dataset root")
    parser.add_argument("--tag", type=str, default="eval",
                        help="Tag name for output folders (e.g., with_gradcam)")
    args = parser.parse_args()

    evaluate_model(
        val_root=args.val_root,
        model_path=args.model,
        mode_tag=args.tag
    )
