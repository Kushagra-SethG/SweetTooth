# SweetTooth: Diabetes Prediction App

## Overview
SweetTooth is a web application built with Streamlit that predicts the risk of diabetes using the Pima Indians Diabetes dataset. The app uses a machine learning model trained in the provided Jupyter notebook (`test-prediction-ipynb.ipynb`).

---

## How the Model Was Trained

The model training process is fully documented in `test-prediction-ipynb.ipynb`. Here is a summary:

1. **Data Loading**: The Pima Indians Diabetes dataset was loaded using pandas.
2. **Exploratory Data Analysis (EDA)**: The dataset was analyzed for class distribution, feature correlations, and missing values.
3. **Handling Imbalance**: The dataset was imbalanced, so SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the classes in the training set.
4. **Feature Scaling**: All features were scaled using `StandardScaler` from scikit-learn. The scaler was saved as `scaler.pkl` for use in the app.
5. **Model Training**:
    - Logistic Regression and SVC (Support Vector Classifier) were trained.
    - Hyperparameters for SVC were tuned (kernel='rbf', C=0.21).
    - The SVC model was selected as the best model and saved as `best_model.pkl`.
6. **Evaluation**: Models were evaluated using accuracy and classification reports.

---

## Features Used
The following features from the dataset are used for prediction:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## How the App (GUI) Works
1. **User Input**: The user enters values for all 8 features in the sidebar or main form.
2. **Scaling**: The input is scaled using the pre-fitted scaler (`scaler.pkl`).
3. **Prediction**: The scaled input is passed to the trained model (`best_model.pkl`) to predict diabetes risk.
4. **Result Display**: The app displays whether the user is at high or low risk for diabetes, along with a summary of the input values.

---

## Methods Explained from `test-prediction-ipynb.ipynb`
- **train_test_split**: Splits the data into training and testing sets.
- **SMOTE**: Balances the training data by oversampling the minority class.
- **StandardScaler**: Scales features to have zero mean and unit variance.
- **LogisticRegression**: A linear model for binary classification.
- **SVC**: Support Vector Classifier, used here with an RBF kernel for non-linear classification.
- **classification_report**: Provides precision, recall, f1-score, and accuracy metrics.
- **joblib.dump**: Saves the trained model and scaler to disk for later use in the app.

---

## How to Run the App
1. Make sure you have Python and Streamlit installed.
2. Place `app.py`, `scaler.pkl`, and `best_model.pkl` in the same directory.
3. (Optional) Create and activate a virtual environment.
4. Install requirements:
   ```bash
   pip install streamlit scikit-learn pandas numpy
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Notes
- The app does not store any user data.
- The model is for educational purposes and not for medical diagnosis.

---

## Credits
- Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Developed by Kushagra-SethG
