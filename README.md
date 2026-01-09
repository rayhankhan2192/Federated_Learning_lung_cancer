# Federated Based Machine Learning for Lung Cancer

# Federated Learning Framework for Interpretable Lung Cancer Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.0%2B-orange.svg)](https://flower.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## 📜 Abstract

This repository contains the official implementation of a privacy-preserving **Federated Learning (FL)** framework designed for the multi-institutional classification of lung computed tomography (CT) scans. Addressing the critical barriers of data silos and model opacity in medical AI, this system enables collaborative training across distributed clients without sharing raw patient data.

A key contribution of this framework is the integration of an **Explainable AI (XAI) Probe** within the federated validation loop. By computing **Grad-CAM++** heatmaps and **Deletion AUC** faithfulness scores in real-time, the system allows for the global monitoring of both diagnostic performance (F1-score) and model trustworthiness (Explainability) simultaneously.

## 🚀 Key Methodological Features

### 1. Federated Architecture
* **Framework:** Built on **Flower (flwr)** and **PyTorch**.
* **Strategy:** Implements a custom `MedicalFLStrategy` (extending `FedAvg`) to manage dynamic client registration, global model aggregation, and history tracking.
* **Privacy Preservation:** Raw data never leaves the local client; only model weight updates are transmitted to the central server.

### 2. Integrated Explainability (XAI)
Unlike standard black-box FL systems, this framework quantifies *why* predictions are made:
* **Grad-CAM++:** Generates high-resolution class activation maps for local validation samples.
* **Deletion AUC (Faithfulness Metric):** Automatically evaluates the reliability of heatmaps by measuring the drop in model confidence when salient pixels are perturbed.
    * *Lower Score (< 0.3):* High Faithfulness (Model relies on the highlighted tumor).
    * *Higher Score (> 0.6):* Low Faithfulness (Model relies on spurious correlations).

### 3. Medical-Grade Preprocessing
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances local contrast to improve nodule visibility in dense lung tissue.
* **Albumentations:** Implements robust augmentation pipelines (Shift, Scale, Rotate, HorizontalFlip) to improve generalization on small local datasets.

### 4. Deep Learning Backbones
Modular design supporting multiple state-of-the-art architectures via `model_factory.py`:
* **DenseNet-121** (Recommended for best balance of F1/Faithfulness)
* **ResNet-50**
* **MobileNetV3**
* **Custom CNN** (Lightweight baseline)

---

## 📂 Repository Structure

| File | Functionality |
| :--- | :--- |
| **`server.py`** | **Central Orchestrator.** Manages the FL lifecycle, aggregates global weights, and logs global convergence metrics (Loss, Accuracy, Deletion AUC). |
| **`client.py`** | **Edge Node.** Handles local training loops, validation inference, and the XAI probing pipeline. |
| **`dataloader.py`** | **Data Ingestion.** Manages `CTScanDataset`, implementing CLAHE preprocessing and patient-wise train/val/test splitting. |
| **`train_eval.py`** | **Training Engine.** Encapsulates the training loop, metric calculation (Sensitivity, Specificity, F1), and TensorBoard logging. |
| **`model_factory.py`** | **Architecture Definitions.** Factory pattern for model instantiation and loss function configuration (CrossEntropy / Focal Loss). |

---

## 🛠️ Installation & Reproduction

### 1. Clone the Repository
```bash
git clone [https://github.com/rayhankhan2192/Federated_Learning_lung_cancer.git](https://github.com/rayhankhan2192/Federated_Learning_lung_cancer.git)
cd Federated_Learning_lung_cancer
```

2.  **Create a Virtual Environment**
    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate (Windows)
    venv\Scripts\activate

    # Activate (Linux/macOS)
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    # Or manually:
    pip install torch torchvision numpy pandas scikit-learn opencv-python albumentations flwr matplotlib seaborn tqdm tensorboard
    ```

## 📊 Dataset Setup

The system expects data to be organized by class folders. The `dataloder.py` script automatically detects classes based on folder names.

```text
/DataSet/Lung-CT-Scan/
├── Benign cases/       # Non-cancerous nodules
│   ├── image_01.jpg
│   └── ...
├── Malignant cases/    # Confirmed carcinomas
│   ├── image_01.jpg
│   └── ...
└── Normal cases/       # Healthy parenchyma
    ├── image_01.jpg
    └── ...
```
🚀 Usage
### 1. Initialize the Server
The server acts as the central aggregator. Start it first and specify the simulation parameters.

````bash
python server.py --rounds 10 --min-clients 2 --model densenet121
````
--rounds: Total number of federated averaging rounds.

--min-clients: Minimum number of clients required to begin training.

--model: Backbone architecture selection. (customcnn, resnet50, densenet121, ViT-ResNet-Hybrid, SwinT-DenseNet-Hybrid).

2. Connect Clients
Launch independent terminals for each participating institution (client). Ensure distinct client-ids for logging purposes.

### Client 1
````bash
python client.py --client-id 1 --data-dir "./DataSet/Lung-CT-Scan" --server-address "localhost:8080"
````

### Client 2
````bash
python client.py --client-id 2 --data-dir "./DataSet/Lung-CT-Scan" --server-address "localhost:8080"
````

### 📈 Outputs & Artifacts
All experimental artifacts are automatically versioned and saved in the Result/ directory.
```text
Server Outputs (Result/FLResult/)
training_curves.png: Visualization of global convergence (Loss, Accuracy, F1-Macro).

best_model_round_X.pth: Serialized weights of the highest-performing global model.

history_round_X.json: comprehensive quantitative logs for post-hoc analysis.

Client Outputs (Result/clientresult/)
xai/: Generated Grad-CAM++ overlays and Deletion AUC scores for local validation samples.

metrics/: Confusion matrices and detailed classification reports (Precision, Recall, F1).

checkpoints/: Local model weights before aggregation.
```