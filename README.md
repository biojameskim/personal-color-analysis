# Personal Color Analysis (using Deep Learning)

**CS4701: Practicum in Artificial Intelligence**

**Authors: James Kim (jjk297), Derek Liu (dtl54), Carla Flores (cmf262)**

## **Overview**

This project explores the use of state-of-the-art deep learning models for personal color analysis using the 12-season Armocromia system. Given a facial image, the goal is to classify users into one of twelve color categories based on skin tone, eye color, and hair tone. We evaluated several model architectures, including a Vision Transformer (ViT), and implemented custom loss functions to support hierarchical classification.

## **Directory Structure**

```
├── inference/
│   ├── joint_loss_inference.py             # Inference script for joint loss model
│   └── hierarchical_softmax_inference.py   # Inference script for hierarchical softmax model
├── models/
│   ├── joint_loss_model.py                 # ViT model with joint loss (season + subtype + full class)
│   └── hierarchical_softmax_model.py       # ViT model with hierarchical softmax loss
├── color_analysis.ipynb                    # Main training notebook with hyperparameter tuning
├── loss_functions.py                       # Joint and hierarchical loss function implementations
├── Loss_Functions.pdf                      # Summary of implemented loss functions
└── README.md                               # Project summary and structure
```

## **Key Features**

* Fine-tuned **ViT-B/16** model on the Deep Armocromia dataset
* Joint classification into **season**, **subtype**, and **12-class label**
* Fine-tuned EfficientNet classification model on the 4 seasons
* Custom **joint loss** and **hierarchical softmax** loss implementations
* Stratified **5-fold cross-validation**, early stopping, and regularization
* Modular design for training, evaluation, and inference

## **Usage**

Run `color_analysis.ipynb` to train and evaluate the model. Modify the config parameters for different loss functions or architectures. Use the inference scripts for downstream prediction.

## **References**

[1] A. Rees, “Colour Analysis Part I: Finding Your Type,” Anuschka Rees, Sep. 24, 2013. [Online]. Available: https://anuschkarees.com/blog/2013/09/24/colour-analysis-part-i-finding-your-type/

[2] “Color Analysis: A Comprehensive Guide,” The Concept Wardrobe. [Online]. Available: https://theconceptwardrobe.com/colour-analysis-comprehensive-guides/what-is-color-analysis

[3] K. Ayyala, “Color Analysis with Deep Learning,” Medium, May 16, 2024. [Online]. Available: https://medium.com/@karthikayyala/color-analysis-with-deep-learning-be9a7a8c2cd3
 
[4] M. Michalski, “Deep Seasonal Color Analysis System (DSCAS),” GitHub, [Online]. Available: https://github.com/mrcmich/deep-seasonal-color-analysis-system/blob/main/Deep_Seasonal_Color_Analysis_System__DSCAS.pdf

[5] PSY222, “Colorinsight: Provide Personalized Color Recommendation Using Face Detection, Segmentation and Image Classification Model,” GitHub, [Online]. Available: https://github.com/PSY222/Colorinsight

[6] Stacchio, L., Paolanti, M., Spigarelli, F., & Frontoni, E. (2024). Deep Armocromia: A novel dataset for face seasonal color analysis and classification. In European Conference on Computer Vision (ECCV) Workshops, (pp. xxx-yyy). Springer.
