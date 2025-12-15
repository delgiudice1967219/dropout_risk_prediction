# Student Dropout Risk Prediction

This project aims to predict the risk of student dropout using various demographic, environmental, and behavioral factors. It leverages machine learning techniques to identify at-risk students, enabling targeted interventions.

## 📌 Usage
To run the analysis, execute the Jupyter Notebook `main.ipynb`. Ensure all dependencies are installed.

```bash
pip install pandas numpy shap pyreadstat matplotlib xgboost imbalanced-learn scikit-learn
```

## 📊 Process Overview

### 1. Data Ingestion
*   **Source**: The dataset is loaded from an SPSS file (`37816-0001-Data.sav`).
*   **Features**: The model utilizes a rich set of features including:
    *   **Demographics**: Age, Sex, Race/Ethnicity, Parent Education, Income.
    *   **Academic Performance**: Grades.
    *   **Environmental Factors**: Drug exposure score, Violence exposure score (Gangs, Weapons, Hate crimes), Crime perception in neighborhood and school.
    *   **Social Support & Behavior**: Isolation score, Avoidance score (avoiding locations due to fear), Fear-based behaviors.

### 2. Preprocessing & Feature Engineering
*   **Target Definitions**: A binary target variable `dropout_risk` (1 = At Risk, 0 = Safe) is derived based on:
    *   Not currently being in school.
    *   No plans for future education.
    *   Chronic truancy (skipping school more than 2 days).
*   **Data Cleaning**:
    *   Missing values are handled using `SimpleImputer` (Median for numerical, Constant for categorical).
    *   Categorical variables are encoded using `OneHotEncoder`.
    *   Numerical variables are scaled using `StandardScaler`.
*   **Class Imbalance**: The dataset is imbalanced (approx. 13.35% at-risk), addressed during modeling.

### 3. Modeling
*   **Algorithm**: **XGBoost Classifier** is used for its efficiency and performance on tabular data.
*   **Imbalance Handling**: The `scale_pos_weight` parameter is tuned to handle the class imbalance effectively.
*   **Hyperparameter Tuning**: `RandomizedSearchCV` is employed to find optimal hyperparameters (Learning rate, Max depth, Number of estimators, etc.) with 5-fold Cross-Validation.

### 4. Evaluation
*   **Metrics**: The model is evaluated using:
    *   **AUC-ROC Score**: To measure discriminative ability.
    *   **F1-Score**: Optimized by finding the best probability threshold using the Precision-Recall curve.
    *   **Confusion Matrix**: To visualize True Positives, False Positives, etc.
    *   **Classification Report**: Precision, Recall, and F1-score for both classes.

### 5. Explainability (SHAP)
*   **Feature Importance**: Global feature importance is visualized to understand which factors contribute most to the model's decisions.
*   **SHAP Values**: SHAP (SHapley Additive exPlanations) summary plots are used to interpret the impact of each feature on the prediction (e.g., how drug exposure or low grades increase dropout risk).

## 🛠 Dependencies
*   `pandas` & `numpy`: Data manipulation.
*   `matplotlib`: Data visualization.
*   `sklearn`: Preprocessing, model selection, and metrics.
*   `xgboost`: Gradient boosting model.
*   `imbalanced-learn`: Tools for imbalanced datasets.
*   `shap`: Model explainability.
*   `pyreadstat`: Loading SPSS data files.
