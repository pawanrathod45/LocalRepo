# Deep Learning Lab Assignment 2  
## Research Paper Implementation with Pre-trained Model  

---

##  Course Information
- **Course Name:** Deep Learning  
- **Lab Title:** Research Paper Implementation with Pre-trained Model  
- **Student Name:** Pawan Rathod  
- **Student ID:** 202402040021  
- **Date of Submission:** 18-02-2026  
- **Group Members:** Om Shinde, Pawan Shinde, Hari Sharma  

---

##  Research Paper Details

- **Title:** Deep Residual Learning for Image Recognition  
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
- **Conference:** CVPR 2016  
- **Paper Link:** https://arxiv.org/abs/1512.03385  

###  Summary
The research paper introduces Residual Networks (ResNet), which address the vanishing gradient problem in deep neural networks using skip connections. These residual connections allow the model to train very deep architectures effectively. The model achieved state-of-the-art performance on ImageNet and CIFAR-10 datasets.

---

##  Dataset Information

- **Dataset Name:** CIFAR-10  
- **Official Link:** https://www.cs.toronto.edu/~kriz/cifar.html  
- **Kaggle Link:** https://www.kaggle.com/c/cifar-10  

### Dataset Description:
- 60,000 color images  
- 10 classes  
- 32×32 image size  
- 50,000 training images  
- 10,000 testing images  

The dataset was loaded using TensorFlow Keras API.

---

##  Model Implementation

- **Model Used:** ResNet50 (Pre-trained on ImageNet)  
- **Transfer Learning:** Applied  
- **Initial Layers:** Frozen  
- **Fine-Tuning:** Last 20 layers unfrozen  
- **Output Layer:** Modified for 10-class classification  

---

##  Hyperparameters

- **Optimizer:** Adam  
- **Learning Rate:** 0.001 (initial), 0.00001 (fine-tuning)  
- **Batch Size:** 32  
- **Epochs:** 5  
- **Loss Function:** Categorical Crossentropy  

---

##  Model Performance

### Classification Metrics:
- **Accuracy:** 35%  
- **Precision:** 0.41  
- **Recall:** 0.35  
- **F1-Score:** 0.31  

### Visualizations Included:
- Accuracy Graph (Training vs Validation)  
- Loss Graph  
- Confusion Matrix  

---

## Performance Comparison with Research Paper

| Metric | Research Paper | My Implementation |
|--------|---------------|-------------------|
| Accuracy | ~92% | 35% |
| Precision | High | 41% |
| Recall | High | 35% |
| F1-Score | High | 31% |

### Discussion:
The original research paper achieved higher accuracy due to:
- Larger input image size (224×224)
- Extensive training epochs
- Powerful computational resources
- Advanced data augmentation techniques

The lower accuracy in this implementation is due to:
- Limited training epochs (5)
- Small input image size (32×32)
- Limited computational resources

---

##  Conclusion

The implementation demonstrates the effectiveness of transfer learning using a pre-trained ResNet50 model. Although the achieved accuracy (35%) is lower than the original research paper, the experiment successfully reproduced the core methodology. Performance can be improved by resizing images to 224×224, increasing training epochs, and applying advanced data augmentation.

---

##  Repository Contents

- Jupyter Notebook (.ipynb file)
- README.md
- Screenshots folder (Training results & graphs)

---

##  Declaration

I confirm that this work is my own and has been completed following academic integrity guidelines.

**Student Name:** Pawan Rathod  
**Student ID:** 202402040021  

---
