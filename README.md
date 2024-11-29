## English

This project involves training a binary classification model to identify the "Attractive" attribute from the CelebA dataset, a large-scale facial attributes dataset containing over 200,000 celebrity images. The focus of the project is to compare the performance of a custom Convolutional Neural Network (CNN) against ResNet50 after transfer learning.

---

### Dataset

The **CelebA dataset** is used for this project. It contains high-quality celebrity images annotated with multiple attributes. The dataset is publicly available on Kaggle:  
[https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

We utilized 20,000 images and focused on the "Attractive" attribute, converting it into a binary classification problem.

---

### Goals

1. **Model Comparison**:  
   - Implement and train a custom CNN architecture from scratch.  
   - Fine-tune the ResNet50 model using transfer learning.  
   - Compare the models based on accuracy, loss, and other evaluation metrics.

2. **Validation Techniques**:  
   - Use K-Fold Cross Validation (5 folds) to ensure reliable performance evaluation.  

3. **Analysis of Results**:  
   - Generate classification reports, confusion matrices, and learning curves for better interpretability.  

---

### Technologies and Frameworks Used

- **Programming Language**: Python  
- **Deep Learning Library**: PyTorch  
- **Other Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  

---

### Key Features

- **Custom Dataset Class**: Efficiently loads and preprocesses images using PyTorch's `Dataset` and `DataLoader`.  
- **Data Augmentation**: Applies a variety of transformations to prevent overfitting.  
- **Mixed Precision Training**: Uses PyTorch's `GradScaler` for improved training performance on GPUs.  
- **Transfer Learning**: Fine-tunes the ResNet50 model to adapt it for the CelebA binary classification task.


## French

Ce projet consiste à entraîner un modèle de classification binaire pour identifier l'attribut "Attractive" à partir du dataset CelebA, un ensemble de données d'attributs faciaux à grande échelle contenant plus de 200 000 images de célébrités. L'objectif principal est de comparer les performances d'un réseau de neurones convolutionnel (CNN) personnalisé avec celles de ResNet50 après un apprentissage par transfert.

---

### Dataset

Le **dataset CelebA** est utilisé pour ce projet. Il contient des images de célébrités de haute qualité annotées avec plusieurs attributs. Le dataset est disponible publiquement sur Kaggle :  
[https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

Nous avons utilisé 20 000 images en nous concentrant sur l'attribut "Attractive", transformé en un problème de classification binaire.

---

### Objectifs

1. **Comparaison des Modèles** :  
   - Implémenter et entraîner une architecture CNN personnalisée.  
   - Ajuster le modèle ResNet50 avec l'apprentissage par transfert.  
   - Comparer les modèles sur la base de la précision, de la perte et d'autres métriques d'évaluation.

2. **Techniques de Validation** :  
   - Utiliser la validation croisée K-Fold (5 plis) pour garantir une évaluation fiable des performances.

3. **Analyse des Résultats** :  
   - Générer des rapports de classification, des matrices de confusion et des courbes d'apprentissage pour une meilleure interprétation.  

---

### Technologies et Frameworks Utilisés

- **Langage de Programmation** : Python  
- **Bibliothèque de Deep Learning** : PyTorch  
- **Autres Bibliothèques** : NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  

---

### Fonctionnalités Clés

- **Classe Dataset Personnalisée** : Charge et prétraite efficacement les images à l'aide de `Dataset` et `DataLoader` de PyTorch.  
- **Augmentation des Données** : Applique diverses transformations pour éviter le surapprentissage.  
- **Entraînement en Précision Mixte** : Utilise `GradScaler` de PyTorch pour améliorer les performances d'entraînement sur GPU.  
- **Apprentissage par Transfert** : Ajuste le modèle ResNet50 pour l'adapter à la tâche de classification binaire du CelebA.  
