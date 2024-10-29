**Compositional Segmentation of Cardiac Images Leveraging Metadata**
**Accepted at WACV 2025 (IEEE/CVF Winter Conference on Applications of Computer Vision)**
Authors: Anonymous for WACV Submission
Supervisors: Muhammad Asad, Martin Benning, Caroline Roney, Greg Slabaugh

Overview
This repository provides the code implementation for our WACV 2025 paper, "Compositional Segmentation of Cardiac Images Leveraging Metadata." Our work presents a novel compositional segmentation approach to efficiently segment cardiac structures by integrating metadata, such as patient demographics and acquisition parameters, to conditionally modulate the segmentation network.

Key Contributions
Compositional Segmentation: A hierarchical approach that performs:

Super-segmentation: Localizes the heart.
Sub-segmentation: Further segments detailed structures within the heart, including left and right ventricles (LV, RV) and myocardium (MYO).
Cross-Modal Feature Integration (CMFI): We introduce a CMFI module to leverage metadata (e.g., scanner type, medical condition, demographic details) as additional context, enhancing segmentation accuracy and robustness.

Evaluation: Our approach was evaluated on two modalities using publicly available datasets:

M&Ms-2 Dataset (MRI)
CAMUS Dataset (Ultrasound)
Results demonstrate state-of-the-art segmentation performance across diverse cardiac imaging modalities.
