# CodeAlpha_Project_IrisFlowerClassification

## Overview

The Iris Flower Classification project aims to build a machine learning model that can accurately classify iris flowers into three species: Setosa, Versicolor, and Virginica, based on their physical measurements. This project serves as an introduction to machine learning concepts, including data preprocessing, model training, evaluation, and prediction.

## Dataset

The dataset used in this project is the well-known Iris dataset, which contains 150 samples of iris flowers. Each sample has four features:

- **Sepal Length (cm)**: The length of the sepal.
- **Sepal Width (cm)**: The width of the sepal.
- **Petal Length (cm)**: The length of the petal.
- **Petal Width (cm)**: The width of the petal.

The target variable consists of three classes representing the species of the iris flower:

- **0**: Setosa
- **1**: Versicolor
- **2**: Virginica

The dataset is available through the Scikit-learn library, making it easy to load and use for classification tasks.

## Methodology

The project follows these key steps:

1. **Data Loading**: The Iris dataset is loaded using the `load_iris()` function from the Scikit-learn library.

2. **Data Exploration**: The feature names and target names are printed, and the first few rows of the dataset are displayed to understand its structure.

3. **Data Preprocessing**:
   - The dataset is split into training and testing sets using `train_test_split()`, with 80% of the data used for training and 20% for testing.
   - The features are standardized using `StandardScaler()` to ensure that they have a mean of 0 and a standard deviation of 1.

4. **Model Training**: A Random Forest classifier is trained on the training data. This ensemble learning method combines multiple decision trees to improve classification accuracy.

5. **Model Evaluation**: The model's performance is evaluated using:
   - **Confusion Matrix**: Displays the number of correct and incorrect predictions for each class.
   - **Classification Report**: Provides precision, recall, F1-score, and support for each class.
   - **Accuracy Score**: Indicates the overall accuracy of the model on the test set.

6. **Making Predictions**: The trained model is used to predict the species of new iris flower measurements.

## Requirements

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run the Code

1. Clone the repository or download the code files.
2. Navigate to the project directory in your terminal.
3. (Optional) Create and activate a virtual environment.
4. Install the required libraries.
5. Run the Python script:

```
python iris_classification.py
```

## Conclusion

This project demonstrates the fundamental concepts of machine learning, including data preprocessing, model training, and evaluation. The Iris dataset serves as an excellent starting point for understanding classification tasks and the application of machine learning algorithms.
