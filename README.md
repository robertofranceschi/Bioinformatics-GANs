# Subtype-GAN 🧬

![example](/images/subtypeGAN_architecture.jpg)
<p align = "center">
Subtype-GAN architecture as in [1] plus a classification head.
</p>

🔗 Check the final [project presentation](presentation.pdf).

## Problem description
**Cancer Suptyping** 🧬 Describes the smaller groups that a type of cancer can be divided into, based on certain characteristics of the cancer cells. These characteristics include how the cancer cells look and specific gene expressions. It is important to know the subtype of a cancer in order to plan treatment and determine prognosis.

**Goal** 🎯 Due to the diversity and complexity of multi-omics data, it is challenging to develop integrated clustering algorithms for tumor molecular subtyping. 
Our objective is to classify lungs cancer subtypes given multi-omics data exploiting the power of Generative adversarial networks.

## Dataset
Data downloaded from GDC portal [4], with manifest files.
- Multi-omics considered: `mRNA`, `miRNA`, `meth` and `images`
- Suptypes considered: `TCGA-LUSC`, `TCGA-LUAD`, `CPTAC-3`, `TCGA-BRCA`, `TCGA-KIRC`.
- Preprocessing (NA, log2, scaler)
  - Feature selection 
  - Dimensionality reduction
- Early integration approach

## Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering
Since images comes without label the method in [2] has been implemented to extract the cancer areas of the images. Consequently the image information are flattened and added as an input.

![example](/images/example_unsupervised_seg.jpg)
<p align = "center">
Example of unsupervised segmentation on pathological images of adenocarcinoma (left) and squamous cell carcinoma (right).
</p>

The following table reports the **classification accuracy** on the test dataset. The best model found is the one that use the SubtypeGAN plus the classification layer. Moreover, the best preprocessing method overall is the ensemble method to choose the best features for each omic.

| Preprocessing | Model | Classifier + Encoder | Only Classifier |
| --- | --- | --- | --- |
| K Best | 0.9792 | 0.9594 | 0.9422 | 
| PCA | 0.9744 | 0.8887 | 0.8758 | 
| Ensemble Method | **0.9805** | 0.9615 | 0.9391 | 

## References
[1] Yang, Hai et al. “Subtype-GAN: a deep learning approach for integrative cancer subtyping of multi-omics data.” Bioinformatics (2021) <br>
[2] Kim, Wonjik et al. “Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering.” IEEE Transactions on Image Processing 29 (2020): 8055-8068. <br>
[3] Wang, S. et al. Comprehensive analysis of lung cancer pathology images to discover tumor shape and boundary features that predict survival outcome (2018) <br>
[4] GDC Dataset: https://portal.gdc.cancer.gov/
