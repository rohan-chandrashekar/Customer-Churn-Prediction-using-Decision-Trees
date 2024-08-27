# Customer Churn Prediction Model

This project focuses on predicting customer churn using historical transaction data from an e-commerce retailer. The dataset captures customer transactions between December 2010 and December 2011, and the goal is to predict whether customers will churn based on their purchasing behavior.

## Problem Statement

The company has been experiencing a decline in repeat customers and high customer churn. Using the transaction data, we aim to build a predictive model that classifies whether a customer is likely to churn or not.

### Churn Definition
For this study:
- **Churn**: A customer who made purchases between Dec 2010 and Aug 2011 but did not return to make any purchases between Sep 2011 and Dec 2011.
- **Not Churn**: A customer who made purchases between Dec 2010 and Aug 2011 and returned to make at least one purchase between Sep 2011 and Dec 2011.

## Key Steps

### 1. Target Variable Creation
- Label customers as **Churn** or **Not Churn** based on their purchasing activity in the defined periods.

### 2. Feature Engineering
- Construct customer-level features from the transaction dataset, such as:
  - **Total Purchase Amount**
  - **Number of Transactions**
  - **Average Transaction Value**
  - **Recency of Purchase**
  - **Customer Tenure**

### 3. Model Construction
- **Decision Tree (ID3 Algorithm)**: Constructed a Decision Tree to classify customers based on whether they will churn or not.
- **SVM (Support Vector Machine)**: Implemented SVM to classify churn with a non-linear kernel.
- **ANN (Artificial Neural Network)**: Used ANN for classification, exploring its capabilities to handle complex patterns in customer behavior.

### 4. Investigation of Oblique Decision Trees
- Explored whether the ID3 algorithm can be adapted to construct oblique decision trees, allowing for more flexible decision boundaries.

### 5. Model Evaluation
- Evaluated all models based on:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **Confusion Matrix**

### 6. Comparative Analysis
- Provided a comparative table highlighting the performance of the three models: Decision Tree, SVM, and ANN.

## Tools & Technologies

- **Python**: Programming language used for implementation.
- **Jupyter Notebook**: For developing and running the project.
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow/keras` (for ANN).

## Results

The project concludes with a detailed comparison of the three models (Decision Tree, SVM, and ANN), demonstrating their respective strengths and weaknesses in predicting customer churn. The comparative analysis provides insight into which model performs best for this dataset.

## Conclusion

Predicting customer churn is critical for improving customer retention and profitability. This project demonstrates how machine learning can help identify at-risk customers and offers valuable insights into different algorithms' effectiveness.

---

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

Ensure you have Python 3.x installed, along with the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

### Running the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/rohan-chandrashekar/customer-churn-prediction.git
   ```
2. Navigate to the project directory and open the Jupyter notebook:
   ```bash
   jupyter notebook Churn.ipynb
   ```
3. Run the notebook to execute the models and view the results.

## Repository Structure

- `Churn.ipynb`: The main Jupyter notebook containing code for data processing, feature engineering, model training, evaluation, and comparison.
- `data.csv`: The dataset containing historical customer transactions (uploaded separately).
- `README.md`: This file.

## Future Work

- Investigate more advanced algorithms such as XGBoost or Random Forest.
- Explore hyperparameter tuning for improved model performance.
- Experiment with additional customer features and time series analysis.

---

This project reflects a real-world scenario where predicting customer churn can significantly impact business strategy. It's designed to highlight skills in data preprocessing, model building, and performance evaluation, making it ideal for academic purposes and showcasing to potential employers.
