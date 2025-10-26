# Predicting Transport Cost for Medical Equipment Deliveries

**Team:** Unsupervised Learners
* Sawant Hrushikesh Rahul (IMT2023619)
* Akshat Mittal (IMT2023606)

---

## 1. Project Overview

This project focuses on building a supervised machine learning model to accurately predict the transport cost for delivering sensitive medical equipment. Reliable cost prediction is essential for optimizing logistics, enabling dynamic pricing, improving resource allocation, and enhancing customer communication.

The project tackles a regression problem using a dataset from a Kaggle competition. We explore various regression models, perform extensive feature engineering, and use hyperparameter optimization to achieve the best possible performance.

## 2. Problem Statement

* **Problem Type:** Supervised Regression
* **Target Variable:** `Transport_Cost`
* **Evaluation Metric:** The primary metric for model evaluation and optimization is the **Root Mean Squared Error (RMSE)**.

## 3. Dataset

The dataset is provided in three files:
* `train.csv`: Contains 5000 samples with 20 columns, including the `Transport_Cost` target variable.
* `test.csv`: Contains 500 samples with 19 columns, for which predictions must be made.
* `sample_submission.csv` (Not in repo, from Kaggle): A template file showing the correct submission format.

## 4. Methodology

Our approach can be broken down into three main stages: preprocessing, feature engineering, and modeling.

### 4.1. Preprocessing

1.  **Imputation:**
    * **Numeric Features:** Missing values (e.g., in `Supplier_Reliability`, `Equipment_Height`) were filled using the **median** computed from the training set.
    * **Categorical Features:** Missing values (e.g., in `Equipment_Type`, `Transport_Method`) were filled using the **mode** (most frequent value) or a placeholder string like 'Unknown'.

2.  **Target Transformation:**
    * The `Transport_Cost` target variable is skewed. To stabilize variance, we applied a **log1p transform** (i.e., $log(1+y)$) before training.
    * All model predictions were transformed back to their original scale using `expm1` (i.e., $e^x - 1$) before calculating the RMSE and generating submissions.

3.  **Encoding:**
    * **Binary Features:** Columns with 'Yes'/'No' values (e.g., `Urgent_Shipping`, `Fragile_Equipment`) were mapped to `1`/`0`.
    * **Categorical Features:** Multi-class categorical columns (e.g., `Equipment_Type`, `Transport_Method`) were converted into numeric features using **One-Hot Encoding** (`pd.get_dummies` with `drop_first=True`).
    * **Exception for CatBoost:** For the CatBoost model, these features were left as strings and passed directly to the model using the `cat_features` argument, leveraging CatBoost's native categorical handling.

### 4.2. Feature Engineering

We created several new features to provide more predictive power to the models:

* **Date-Based Features:** Parsed `Order_Placed_Date` and `Delivery_Date` to create:
    * `Delivery_Time_Days`: The number of days between order and delivery.
    * `Order_Year`, `Order_Month`, `Order_Day_of_Week`.
    * `Is_Weekend`.
* **Physical Features:** Combined columns to create domain-specific features:
    * `Equipment_Area`: `Equipment_Height` * `Equipment_Width`.
    * `Value_per_Weight`: `Equipment_Value` / `Equipment_Weight`.
* **Interaction Features:** Created multiplicative features to capture combined effects, such as `Urgent_x_CrossBorder` and `Fragile_x_Rural`.

### 4.3. Models and Hyperparameter Tuning

We evaluated six different regression models. For robust tuning, we used the **Optuna** library with a TPE sampler.

1.  **Linear Regression (Baseline):** Standard Ridge and Lasso models.
2.  **Random Forest Regressor:** Tuned with Optuna using 5-fold cross-validation.
3.  **AdaBoost Regressor:** Tuned with Optuna using a static 80/20 validation split.
4.  **Gradient Boosting Regressor:** A standard implementation from scikit-learn.
5.  **CatBoost Regressor:** Tuned with Optuna, using its native categorical feature support.
6.  **XGBoost Regressor (Winning Model):** Tuned with Optuna over 300 trials using 5-fold cross-validation and inner-fold early stopping for robust performance.

## 5. Results

The **XGBoost Regressor** provided the best performance, achieving the lowest (best) score on the private Kaggle leaderboard.

| Model | Kaggle Score |
| :--- | :--- |
| **XGBoost** | **38,767,56604.375** |
| Random Forest | 46,849,89841.144 |
| CatBoost | 48,497,82813.369 |
| Gradient Boosting | 53,357,18020.125 |
| Ada Boost | 56,681,69936.815 |
| Linear regression | 59,813,95947.328 |


The superior performance of XGBoost is attributed to its advanced regularization, robust hyperparameter tuning with 5-Fold CV, and efficient training with early stopping to prevent overfitting.

## 6. File Descriptions

* **`train.csv`**: The main training dataset (5,000 samples) with all features, including the `Transport_Cost` target variable.
* **`test.csv`**: The test dataset (500 samples) without the target variable. This is used to generate the final predictions.
* **`eda.ipynb`**: (Exploratory Data Analysis) A Jupyter Notebook containing the initial investigation of the dataset, including visualizing feature distributions, checking for missing values, and identifying correlations.
* **`model_*.py` scripts**: A set of Python scripts, each containing the specific implementation and tuning for a single regression model (e.g., `model_xgboost.py`, `model_adaboost.py`).
* **`Final.ipynb`**: This is the main Jupyter Notebook that consolidates the entire project. It contains the final data preprocessing pipeline, feature engineering, and code to run the models (likely by leveraging the `.py` scripts) and generate the winning submission file.
* **`submission_xgb_v3_fine_tuned.csv`**: The final, best-scoring submission file, generated by the tuned XGBoost model.
* **`Machine_Learning_Assignment_1 (1).pdf`**: The final project report detailing the problem statement, methodology, all model pipelines, and the final results and comparisons.
* **`README.md`**: This file. It serves as the homepage for the repository.
* **`requirements.txt`**: A text file listing the necessary Python libraries (e.g., `pandas`, `xgboost`, `optuna`) to run the notebooks and scripts.

## 7. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Hrushi199/Machine-Learning-Assignment.git](https://github.com/Hrushi199/Machine-Learning-Assignment.git)
    cd Machine-Learning-Assignment
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. You can install the required libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Or, install the main ones manually:
    ```bash
    pip install pandas numpy scikit-learn xgboost catboost optuna matplotlib jupyter
    ```

3.  **Run the analysis:**
    * All necessary data and notebooks are in the root directory.
    * Launch Jupyter Notebook:
        ```bash
        jupyter notebook
        ```
    * First, you can open **`eda.ipynb`** to see the initial data exploration.
    * Next, open and run the cells in **`Final.ipynb`**. This is the main script that will perform all preprocessing, train the models, and generate the final submission file: `submission_xgb_v3_fine_tuned.csv`.
