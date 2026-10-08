# pancreatic-cancer-ct-segmentation

**# Pancreatic Cancer CT Segmentation Using Deep Learning**



**## 📌 Project Overview**



**This project focuses on the analysis and segmentation of pancreatic regions from Computed Tomography (CT) images using deep learning techniques.**



**The main objective is to develop a deep learning-based approach that can automatically identify and segment the pancreas from CT images. The project is part of a research study on computer-aided analysis of pancreatic cancer using medical images.**



**The project uses the \*\*Medical Segmentation Decathlon (MSD) Task 07 – Pancreas\*\* dataset for experimentation and evaluation.**



**> \*\*Note:\*\* The dataset is not included in this GitHub repository due to its large size. It is downloaded and stored locally for research purposes.**



**---**



**## 🎯 Objectives**



**- Study pancreatic CT images and their characteristics.**

**- Preprocess CT images for deep learning.**

**- Perform pancreas segmentation using deep learning techniques.**

**- Experiment with suitable segmentation architectures.**

**- Evaluate segmentation performance using appropriate metrics.**

**- Visualize and analyze segmentation results.**

**- Investigate approaches that may contribute to improved computer-aided pancreatic cancer analysis.**



**---**



**## 📊 Dataset**



**### Medical Segmentation Decathlon — Task 07: Pancreas**



**The project uses the \*\*MSD Task 07 Pancreas\*\* dataset.**



**The dataset contains abdominal CT images along with corresponding segmentation labels.**



**### Dataset Contents**



**The dataset includes:**



**- CT images**

**- Segmentation masks/labels**

**- Training and testing data**

**- Dataset metadata**



**### Dataset Location**



**The dataset is stored \*\*outside this GitHub repository\*\*.**



**Example local structure:**



**```text**

**MSD\_Task07\_Pancreas/**

**├── imagesTr/**

**├── imagesTs/**

**├── labelsTr/**

**└── dataset.json**

**```**



**The dataset files are excluded from Git using `.gitignore`.**



**---**



**## 🧠 Methodology**



**The research workflow will generally consist of the following stages:**



**```text**

**CT Images**

&#x20;    **↓**

**Data Preprocessing**

&#x20;    **↓**

**Image/Volume Preparation**

&#x20;    **↓**

**Deep Learning Model**

&#x20;    **↓**

**Pancreas Segmentation**

&#x20;    **↓**

**Post-processing**

&#x20;    **↓**

**Evaluation**

&#x20;    **↓**

**Visualization \& Analysis**

**```**



**The exact model architecture and methodology will be finalized during the research process.**



**---**



**## 🛠️ Technologies**



**The project is being developed using:**



**- Python**

**- PyTorch**

**- NumPy**

**- OpenCV**

**- NiBabel**

**- Matplotlib**

**- Jupyter Notebook**



**Additional libraries may be added as the research progresses.**



**---**



**## 📁 Project Structure**



**```text**

**pancreatic-cancer-ct-segmentation/**

**│**

**├── README.md**

**├── .gitignore**

**│**

**├── src/**

**│   ├── preprocessing/**

**│   ├── segmentation/**

**│   ├── evaluation/**

**│   └── visualization/**

**│**

**├── notebooks/**

**│**

**├── configs/**

**│**

**├── results/**

**│**

**└── docs/**

**```**



**The structure will be updated as the project develops.**



**---**



**## 📈 Evaluation**



**The segmentation models will be evaluated using suitable medical image segmentation metrics, such as:**



**- Dice Similarity Coefficient (DSC)**

**- Intersection over Union (IoU)**

**- Precision**

**- Recall**

**- Hausdorff Distance**



**The final evaluation metrics will depend on the selected methodology.**



**---**



**## 🔬 Research Status**



**\*\*Current Status:\*\* Research and development**



**### Completed**



**- \[x] Downloaded MSD Task 07 Pancreas dataset**

**- \[x] Created GitHub repository**

**- \[x] Configured `.gitignore`**

**- \[ ] Dataset exploration**

**- \[ ] Data preprocessing**

**- \[ ] Baseline segmentation model**

**- \[ ] Model training**

**- \[ ] Model evaluation**

**- \[ ] Result visualization**

**- \[ ] Model comparison**

**- \[ ] Research documentation**



**---**



**## ⚠️ Dataset Notice**



**The raw medical imaging dataset is \*\*not included in this repository\*\*.**



**Large CT volumes and segmentation files are excluded using `.gitignore`.**



**Users interested in reproducing the experiments should obtain the appropriate dataset from its official source and configure the local dataset path accordingly.**



**---**



**## 👩‍💻 Author**



**\*\*Aleena Antony\*\***



**B.Tech Computer Science and Engineering**



**---**



**## 📌 Disclaimer**



**This project is intended for \*\*academic and research purposes\*\*. It is not intended to provide medical diagnosis or replace professional medical judgment.**

