# Pancreatic Cancer CT Segmentation Using Deep Learning

A research project focused on **pancreatic CT image segmentation using deep learning techniques**.

---

## 📌 Overview

Pancreatic cancer is a serious disease where early detection and accurate analysis can support clinical decision-making. Medical image segmentation is an important step in computer-aided analysis because it helps identify and isolate anatomical structures from medical images.

This project investigates **deep learning-based segmentation of the pancreas from CT images**.

The project uses the **Medical Segmentation Decathlon (MSD) Task 07 – Pancreas** dataset for experimentation and evaluation.

> **Dataset Notice:** The MSD Task 07 Pancreas dataset is **not included in this repository**. The dataset is stored locally and excluded from Git using `.gitignore`.

---

## 🎯 Objectives

- Analyze pancreatic CT images.
- Preprocess CT images for deep learning.
- Develop a pancreas segmentation pipeline.
- Experiment with suitable deep learning segmentation architectures.
- Evaluate segmentation performance using appropriate metrics.
- Visualize and analyze segmentation results.
- Investigate approaches that may improve computer-aided pancreatic image analysis.

---

## 📊 Dataset

### Medical Segmentation Decathlon — Task 07: Pancreas

This project uses the **MSD Task 07 Pancreas** dataset.

The dataset contains abdominal CT volumes and corresponding segmentation labels.

### Dataset Contents

The dataset includes:

- CT image volumes
- Segmentation masks
- Training and testing data
- Dataset metadata

### Local Dataset Structure

The dataset is stored outside this GitHub repository.

```text
MSD_Task07_Pancreas/
│
├── imagesTr/
├── imagesTs/
├── labelsTr/
└── dataset.json
```

The raw dataset files are excluded from version control using `.gitignore`.

---

## 🧠 Research Workflow

The current research workflow is planned as follows:

```text
CT Images
    │
    ▼
Data Preprocessing
    │
    ▼
Image / Volume Preparation
    │
    ▼
Deep Learning Model
    │
    ▼
Pancreas Segmentation
    │
    ▼
Post-processing
    │
    ▼
Evaluation
    │
    ▼
Visualization & Analysis
```

The exact model architecture and methodology will be finalized as the research progresses.

---

## 🛠️ Technologies

The project is being developed using:

- **Python**
- **PyTorch**
- **NumPy**
- **OpenCV**
- **NiBabel**
- **Matplotlib**
- **Jupyter Notebook**

Additional libraries will be added as required during development.

---

## 📁 Repository Structure

```text
pancreatic-cancer-ct-segmentation/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── evaluation/
│   └── visualization/
│
├── notebooks/
│
├── configs/
│
├── results/
│
└── docs/
```

The repository structure will evolve as the project progresses.

---

## 📈 Evaluation

The segmentation models will be evaluated using appropriate medical image segmentation metrics, including:

- **Dice Similarity Coefficient (DSC)**
- **Intersection over Union (IoU)**
- **Precision**
- **Recall**
- **Hausdorff Distance**

The final evaluation methodology will be determined during the research process.

---

## 🔬 Research Progress

| Task | Status |
|------|--------|
| MSD Task 07 Pancreas dataset downloaded | ✅ Completed |
| GitHub repository created | ✅ Completed |
| `.gitignore` configured | ✅ Completed |
| Dataset exploration | 🔄 In Progress |
| Data preprocessing | ⬜ Pending |
| Baseline segmentation model | ⬜ Pending |
| Model training | ⬜ Pending |
| Model evaluation | ⬜ Pending |
| Result visualization | ⬜ Pending |
| Model comparison | ⬜ Pending |
| Research documentation | ⬜ Pending |

---

## ⚠️ Dataset & Privacy Notice

The raw CT images and segmentation labels are **not stored in this GitHub repository**.

Large medical imaging files such as `.nii` and `.nii.gz` are excluded through `.gitignore`.

This repository contains the **research code, documentation, configurations, notebooks, and selected results**, rather than the raw dataset.

---

## 👩‍💻 Author

**Aleena Antony**  
B.Tech Computer Science and Engineering

---

## 📌 Disclaimer

This project is intended for **academic and research purposes only**.

It is not intended to provide medical diagnosis, treatment recommendations, or to replace the judgment of qualified healthcare professionals.
