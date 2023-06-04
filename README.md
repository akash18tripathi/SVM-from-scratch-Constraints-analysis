# SVM-from-scratch-Constraints-analysis

This repository provides a Python implementation of Support Vector Machines (SVM) from scratch using a quadratic solver like CXPY. The implementation includes both soft margin and hard margin SVM algorithms.

If you are unable to render the notebook on github, you can view it [here](https://nbviewer.org/github/akash18tripathi/Gaussian-Naive-Bayes-From-Scratch/blob/main/Gaussian%20Naive%20Bayes.ipynb)

## What Jupyter Notebook has to offer?

This Jupyter Notebook provides an in-depth exploration of Support Vector Machines (SVM) for classification tasks. It covers various topics and implementations related to SVM. The notebook includes the following sections:

## 1) Hard Margin SVM Implementation

This section focuses on implementing the hard margin SVM algorithm from scratch. It includes the necessary steps and code to train a hard margin SVM model on a synthetic dataset. The implementation utilizes the quadratic solver approach to find the optimal hyperplane.

## 2) Exploring Problems with Hard Margin SVM

This section delves into the limitations of hard margin SVM when the data is not linearly separable or when outliers are present. It discusses the challenges faced by the hard margin SVM and the need for a more flexible approach.

## 3) Soft Margin SVM

In this section, we introduce the concept of soft margin SVM, which allows for misclassifications and a more flexible decision boundary. It covers the formulation of the soft margin SVM objective function and the training process. The implementation utilizes a quadratic solver to find the optimal solution.

## 4) Handwritten Derivations for Soft Margin SVM

To enhance understanding, this section provides step-by-step handwritten derivations of the mathematical formulas involved in soft margin SVM. The derivations clarify the intuition behind the soft margin SVM algorithm and its various components.

## 5) Experimentation and Analysis with Different Values of C

This section focuses on experimenting with different values of the regularization parameter C in soft margin SVM. It explores the impact of C on the decision boundary, margin, and classification performance. It includes visualizations and analysis of the results.

## 6) MNIST Dataset and SKLEARN Implementation

Finally, this section utilizes the well-known MNIST dataset for digit recognition. It demonstrates the usage of scikit-learn's SVM implementation with different kernels (linear, polynomial, RBF) and compares their performance. It provides insights into the applicability of SVM to real-world datasets.

Feel free to explore and run each section of the notebook to gain a comprehensive understanding of SVM and its various aspects.


## Concepts

### Support Vector Machines (SVM)

Support Vector Machines are powerful supervised learning models used for classification and regression tasks. SVMs aim to find the optimal hyperplane that separates data points of different classes with the maximum margin. SVMs work by transforming the data into a higher-dimensional feature space and finding the hyperplane that maximizes the margin between the support vectors (data points closest to the decision boundary).

### Hard Margin SVM

Hard margin SVM assumes that the data is linearly separable without any errors. It tries to find the hyperplane that separates the data points with no misclassifications. The objective of hard margin SVM can be defined as follows:

minimize: 1/2 * ||w||^2

subject to: yi * (w * xi + b) >= 1 for all training examples (xi, yi)

Where:
- xi is the input vector
- yi is the corresponding class label (+1 or -1)
- w is the weight vector
- b is the bias term

### Soft Margin SVM

Soft margin SVM is a more flexible approach that allows for misclassifications. It introduces a slack variable to allow some data points to be within the margin or even on the wrong side of the decision boundary. The objective of soft margin SVM can be defined as follows:

minimize: 1/2 * ||w||^2 + C * Σξ

subject to: yi * (w * xi + b) >= 1 - ξ for all training examples (xi, yi)
            ξ >= 0

Where:
- C is a hyperparameter that controls the trade-off between maximizing the margin and minimizing the misclassifications
- ξ is the slack variable

### Hyperparameter C

The C parameter in soft margin SVM determines the trade-off between maximizing the margin and minimizing misclassifications. A smaller C allows a larger margin with potential misclassifications, while a larger C imposes stricter penalties, resulting in a smaller margin and potential overfitting.

## SVM Kernels

Support Vector Machines (SVM) offer various kernel functions to model complex decision boundaries. Each kernel transforms the input data to a higher-dimensional space to enable nonlinear classification. The following are commonly used SVM kernels:

### Linear Kernel

The linear kernel creates a straight-line decision boundary in the original feature space.

### Polynomial Kernel

The polynomial kernel maps the input features into a higher-dimensional space using polynomial functions. It allows for curved decision boundaries.

### RBF (Radial Basis Function) Kernel

The RBF kernel calculates the similarity between data points in the feature space. It uses a Gaussian function to create non-linear decision boundaries that are flexible and can capture complex relationships between data points.

### Sigmoid Kernel

The sigmoid kernel transforms the input features using a sigmoid function. It can be useful when dealing with binary classification problems but is less commonly used compared to other kernels.

Each kernel has its own characteristics and is suitable for different types of data and problem domains. The choice of kernel depends on the underlying data distribution and the desired complexity of the decision boundary.


## Quadratic Solver used!

The solver library CXPY is utilized to solve the quadratic optimization problem involved in training SVM.

## Jupyter Notebook

The Jupyter Notebook in this repository contains the implementation of Gaussian Naive Bayes using Python. It includes the necessary code and explanations to understand the algorithm and apply it to classification tasks.

## Contributing

Contributions are welcome and encouraged!
