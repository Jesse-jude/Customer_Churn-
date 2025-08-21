# Telco Customer Churn Predictor

This project provides an end-to-end machine learning pipeline to predict customer churn for a telecommunications company. The model is built using Python and popular data science libraries, and the final trained model is saved for easy inference on new customer data.


*Correlation heatmap of numerical features*

---

## 🚀 Features

- **Complete ML Pipeline:** From data loading and cleaning to model training and evaluation.
- **In-depth EDA:** Comprehensive Exploratory Data Analysis with visualizations to understand data distributions and feature relationships.
- **Handles Imbalanced Data:** Uses the SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the target variable.
- **Model Comparison:** Trains and evaluates three different classification models (Decision Tree, Random Forest, XGBoost) using cross-validation.
- **Ready for Inference:** Saves the best-performing model (`RandomForestClassifier`) and the necessary data encoders using `pickle`.
- **Example Predictions:** Includes a clear example of how to load the saved model and make predictions on new, unseen customer data.

---

## 📊 Dataset

This project uses the **"Telco Customer Churn"** dataset, which is publicly available. It contains customer account information, demographic data, and services they have signed up for.

- **Source:** Kaggle - Telco Customer Churn
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Customer_Churn_Predictor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    Place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the root directory of the project.

---

## ⚙️ Usage

To run the entire pipeline (data processing, training, evaluation, and model saving), simply execute the main script:

```bash
python churn.py
```

This will:
- Perform all the data analysis and print out model evaluation metrics.
- Generate and save two files:
  - `customer_churn_model.pkl`: The trained Random Forest model and feature names.
  - `encoders.pkl`: The `LabelEncoder` objects used for categorical features.

---

##  workflow Pipeline

The script follows a structured machine learning workflow:

1.  **Data Loading & Cleaning:** The dataset is loaded, and initial cleaning steps are performed, such as dropping unnecessary columns and handling missing values in `TotalCharges`.
2.  **Exploratory Data Analysis (EDA):** Visualizations like histograms, box plots, and count plots are used to explore the data's characteristics. A correlation heatmap is generated for numerical features.
3.  **Feature Engineering:** Categorical features are converted into a numerical format using `sklearn.preprocessing.LabelEncoder`. The encoders are saved for later use.
4.  **Model Preparation:**
    - The data is split into features (X) and target (Y).
    - The dataset is divided into training (80%) and testing (20%) sets.
    - **SMOTE** is applied to the *training data* to correct for the imbalanced distribution of the 'Churn' class.
5.  **Model Training & Selection:**
    - Three models are trained and compared using 5-fold cross-validation on the balanced training data: Decision Tree, Random Forest, and XGBoost.
    - Random Forest was selected as the final model based on its robust performance.
6.  **Final Evaluation:** The final model is evaluated on the unseen test set. Performance is measured using:
    - Accuracy Score
    - Confusion Matrix
    - Classification Report (Precision, Recall, F1-Score)
7.  **Serialization:** The trained model and the label encoders are serialized to disk using `pickle`, making them available for future predictions without retraining.

---

## 🔮 Making a Prediction

The saved `customer_churn_model.pkl` and `encoders.pkl` files can be used to predict churn for a new customer. The final section of `churn.py` demonstrates this process.

Here is a code snippet showing how to do it:

```python
import pandas as pd
import pickle

# 1. Load the model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    loaded_model = model_data["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# 2. Create a DataFrame for the new customer
new_customer = pd.DataFrame([{
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
    'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
    'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85, 'TotalCharges': 29.85
}])

# 3. Encode categorical features
for column, encoder in encoders.items():
    new_customer[column] = encoder.transform(new_customer[column])

# 4. Make the prediction
prediction = loaded_model.predict(new_customer)
prediction_proba = loaded_model.predict_proba(new_customer)

print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability (No Churn, Churn): {prediction_proba}")
