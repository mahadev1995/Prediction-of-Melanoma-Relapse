# VisioMel Challenge: Predicting Melanoma Relapse

This repository contains code and report for seminar project on predicting melanoma relapse using digital microscopic slides and clinical metadata. Our work compares the performance of one machine learning approach and four deep learning approaches, with results indicating that deep learning models outperform machine learning models.
###### Schematic diagram of overall relapse prediction process
<img src="/images/aim.svg" width="512" height="270"  title="Aim" />

Our approach employs a two-stage training strategy. In the first stage, we train our models using only image data. In the second stage, we fine-tune our models by incorporating clinical metadata to improve their predictive power. Additionally, we enhance the performance of our models by transforming Whole Slide Images (WSIs) into stitched tiles.
###### Schematic diagram of tile stitching process
<img src="/images/Tilling.svg" width="512" height="220"  title="Tilling" />

Our findings demonstrate that deep learning methods, particularly when combined with a two-stage training strategy and tile stitching, can improve the accuracy of melanoma relapse prediction.

# Model Architecture

| Whole Slide Model (Single Stage) | Patch Based Model (Single Stage) |
| ------ | ------ |
|    <img src="/images/page0.png" width="300" height="400" title="" />    |     <img src="/images/page5.png" width="300" height="420" />   |

# Folder Descriptions
> - **Base_Model** directory contains code for Base model (Machine Learning only using meta data)
> - **Patch_Based_Model** directory contains code for Patch based model (Deep Learning model using patches from WSI and meta data)
> - **WSI_Model** directory contains code for  Whole slide model (Deep Learning model using both WSI and meta data)
> - **Documentation** directory contains latex code for the report as well as the final presentation file



