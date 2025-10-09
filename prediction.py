import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = r"E:\Python\Research\LungCancerFL\Federated_Learning_lung_cancer"
DATA_DIR     = r"E:\Python\Research\LungCancerFL\Federated_Learning_lung_cancer\DataSet\Lung-CT Scan"
IMG_SIZE     = (224, 224)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sys.path.append(PROJECT_ROOT)
from models.model_factory import get_model  

def medical_preprocess_gray(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE + resize + normalization for CT scans."""
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image.astype(np.uint8)

transform = A.Compose([
    A.Resize(IMG_SIZE[1], IMG_SIZE[0]),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2()
])

def prepare_tensor(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    img = medical_preprocess_gray(img)
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
    return tensor, img

def load_model(model_path: str, num_classes: int):
    """Load checkpoint safely into model."""
    model = get_model("customcnn", num_classes=num_classes, pretrained=False, dropout_rate=0.5).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    clean_sd = OrderedDict()
    for k, v in state_dict.items():
        clean_sd[k.replace("module.", "", 1) if k.startswith("module.") else k] = v

    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    return model

def predict_and_save(model, img_path: str, class_names, out_dir: str):
    tensor, disp_img = prepare_tensor(img_path)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        conf = probs[pred_idx] * 100

    # Save output with prediction title
    plt.imshow(disp_img, cmap="gray")
    plt.title(f"Predicted = {pred_class} ({conf:.2f}%)")
    plt.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(img_path)
    save_path = os.path.join(out_dir, base_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved prediction image: {save_path}")
    return pred_class, conf, probs

def main():
    model_path = r"E:\Python\Research\LungCancerFL\Federated_Learning_lung_cancer\Result\FLResult\fl_results_20251008_205502\last_global_model.pth"
    img_path   = r"image copy 3.png"
    out_dir    = r"E:\Python\Research\LungCancerFL\Federated_Learning_lung_cancer\predicted_outputs"

    class_names = ["Benign case", "Malignant case", "Normal case"]

    model = load_model(model_path, num_classes=len(class_names))
    pred_class, conf, probs = predict_and_save(model, img_path, class_names, out_dir)

    print("\nClass probabilities:")
    for i, c in enumerate(class_names):
        print(f"  {c}: {probs[i]*100:.2f}%")


if __name__ == "__main__":
    main()
