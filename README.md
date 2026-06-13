# Global House Purchase Decision Prediction

A machine learning project that predicts whether a user will decide to purchase a property based on financial, demographic, and property-related features. The project explores multiple classification models and evaluates their performance after careful preprocessing and feature engineering.

## Problem Statement

Given a dataset containing housing attributes, financial indicators, and user profiles, the goal is to predict the binary decision:

- Buy (1)
- Not Buy (0)

## Dataset

- 200,000 records
- Features include:
  - Property characteristics (size, rooms, bathrooms, age, type)
  - Financial features (salary, loan amount, monthly expenses)
  - Risk indicators (crime rate, legal cases, connectivity score)
  - User satisfaction and neighborhood ratings
  - Target variable: `decision`

## Feature Engineering

- Property age derived from construction year
- Interaction-based financial features (later removed due to leakage risk)
- One-hot encoding for categorical variables:
  - country
  - city
  - property_type
  - furnishing_status

## Data Preprocessing

- Missing value imputation (median for numeric features)
- Standard scaling for numerical variables
- One-hot encoding for categorical variables
- SMOTE applied to handle class imbalance
- Train-test split: 80/20

## Important Note on Data Leakage

During experimentation, several features were identified as causing leakage and were removed:

- emi_to_income_ratio
- price_per_sqft
- emi_burden
- down_payment

This significantly improved model validity and reduced inflated performance metrics.

## Models Evaluated

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Multi-layer Perceptron (Neural Network)

## Results

After preprocessing and leakage removal, models achieved the following performance:

- Neural Network: best overall performance (~0.998 accuracy, ~0.995 F1)
- Gradient Boosting: strong performance with high ROC-AUC
- Decision Tree: competitive but slightly overfit
- Random Forest: stable but lower recall compared to boosting models
- Logistic Regression: strong baseline but underfits nonlinear structure
- AdaBoost: weakest performance, struggled with noisy feature space

## Key Insights

- Leakage features significantly inflated early results
- Ensemble methods (especially Gradient Boosting) performed best on structured tabular data
- Neural network achieved highest accuracy but required careful tuning
- Feature engineering had a larger impact than model complexity

## Technologies Used

- Python
- Pandas / NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn
- Jupyter Notebook

## Future Improvements

- Hyperparameter optimization with Optuna
- Feature selection via SHAP or permutation importance
- Deployment as a web app (Flask or FastAPI)
- Better calibration of probability outputs

## Author

Built as part of an academic machine learning project.
