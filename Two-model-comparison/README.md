# Plant Disease Classification: CNN vs ResNet-50

## 📌 Project Overview

This project focuses on **plant disease image classification** using two different deep learning architectures: a custom **5-layer Convolutional Neural Network (CNN)** and a **fine-tuned ResNet-50**.

The primary objective is to compare the classification performance of both models on the same **29-class plant disease dataset** and determine which architecture provides better results.

## 🔬 Models Compared

### Custom CNN

A custom 5-layer CNN was trained from scratch for plant disease classification. The model was designed as a relatively lightweight architecture to evaluate how a custom CNN performs against a deeper pretrained network.

### ResNet-50

ResNet-50 was initialized with **ImageNet pretrained weights** and fine-tuned for the plant disease classification task. The final classification layer was adapted to predict the 29 target classes.

## 📊 Performance Comparison

Both models were evaluated using the same test dataset and the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score

### Custom CNN

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **[0.9720]** |
| Precision | **[0.9742]** |
| Recall    | **[0.9720]** |
| F1 Score  | **[0.9719]** |

### ResNet-50

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **[0.9971]** |
| Precision | **[0.9971]** |
| Recall    | **[0.9971]** |
| F1 Score  | **[0.9971]** |


## 📈 Results

Based on the evaluation metrics, **[ADD BEST MODEL NAME]** achieved the better overall classification performance.

**Best Model:** [ADD MODEL NAME]

**Accuracy:** [ADD ACCURACY]

The comparison demonstrates the difference between a custom CNN trained from scratch and a deeper model benefiting from pretrained ImageNet features.

## 💡 Conclusion

The experiment compares two different approaches to plant disease classification:

* A custom CNN provides a simpler, lightweight architecture.
* ResNet-50 benefits from a deeper architecture and pretrained visual features.
* The final model selection was based on performance across multiple evaluation metrics.

The **best-performing model was selected for deployment** as the plant disease classification application.

## 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Streamlit
* NumPy
* Pillow

## 👤 Author

**[Shibusri Behera]**

GitHub: [(https://github.com/Shibu0div)]

Email: [shibusribehera@gmail.com]


