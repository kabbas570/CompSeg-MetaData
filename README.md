# Compositional Segmentation of Cardiac Images Leveraging Metadata

## [Accepted at WACV 2025 (IEEE/CVF Winter Conference on Applications of Computer Vision)](https://wacv2025.thecvf.com/)

You can view or download the paper [HERE](https://arxiv.org/abs/2410.23130).


# Overview:
This repository provides the code implementation for our paper, "Compositional Segmentation of Cardiac Images Leveraging Metadata." Our work presents a novel compositional segmentation approach to efficiently segment cardiac structures by integrating metadata, such as patient demographics and acquisition parameters, to modulate the segmentation network conditionally.

![image](https://raw.githubusercontent.com/kabbas570/CompSeg-MetaData/09d175b70f1e6c1a4b33c172531754a7eb72f4f1/figures/arch.png)

## Key Contributions:
***Compositional Segmentation:*** A hierarchical approach that performs super and sub-segmentation. 

***Super-segmentation:*** Localizes the heart.

***Sub-segmentation:***  Further segments detail structures within the heart, including the left and right ventricles (LV, RV) and the myocardium (MYO).

***Cross-Modal Feature Integration (CMFI):*** We introduce a CMFI module to leverage metadata (e.g., scanner type, medical condition, demographic details) as additional context, enhancing segmentation accuracy and robustness.

## Evaluation:
Our approach was evaluated on two modalities using publicly available datasets:

***M&Ms-2 Dataset (MRI)***

***CAMUS Dataset (Ultrasound)***

Results demonstrate state-of-the-art segmentation performance across diverse cardiac imaging modalities.

# Training Steps

## Segmentation Model 

The segmentaton models do not apply any activation on logits, so add activation after the last 1x1 conv layer or incorporate it in your loss function. Change Num_Classes to your (Segmentation class + 1).(1 for background). We have provided both versions

[Model_with_CMFI](https://github.com/kabbas570/CompSeg-MetaData/blob/264829f6a4532e9da65117ec52a43b1e998aef1e/Compsitional-Model/Model_with_CMFI.py)

[Model_without_CMFI](https://github.com/kabbas570/CompSeg-MetaData/blob/main/Compsitional-Model/Model_without_CMFI.py)



# Citation
@inproceedings{author2025compsegmetadata,

  title={Compositional Segmentation of Cardiac Images Leveraging Metadata},
  
  author={Abbas Khan, Muhammad Asad, Martin Benning, Caroline Roney, and Greg Slabaugh},
  
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  
  year={2025}
}

