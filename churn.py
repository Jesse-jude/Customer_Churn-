# --- 1. Import Libraries ---
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting and visualization
import seaborn as sns  # For statistical data visualization
from sklearn.preprocessing import LabelEncoder  # To encode categorical labels into numbers
from imblearn.over_sampling import SMOTE  # To handle imbalanced datasets by oversampling
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data and cross-validation
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from xgboost import XGBClassifier  # XGBoost model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For model evaluation
import pickle  # For saving and loading the trained model and encoders

# --- 2. Load and Inspect Data ---
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape  # Display the dimensions (rows, columns) of the DataFrame
df.shape  # Displaying shape again for interactive analysis

# Dropping CustomerID as it is not required for modelling
df = df.drop(columns=["customerID"])

pd.set_option("display.max_columns", None)  # Configure pandas to display all columns
df.head(2)  # Display the first 2 rows to get a glimpse of the data

#printing all unique values in all the columns
numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]  # Define numerical columns

for col in df.columns:
    if col not in numerical_features_list:
            print(col, df[col].unique())
            print(""*50)

# --- 3. Data Cleaning and Preprocessing ---
print(df.isnull().sum())  # Check for missing values (NaN) in each column

df[df["TotalCharges"]==" "]  # Identify rows where 'TotalCharges' is a blank space
len(df[df["TotalCharges"]==" "])  # Count the number of rows with blank 'TotalCharges'

# Replace blank spaces in 'TotalCharges' with "0.0" and convert the column to a numeric type
df["TotalCharges"] = df["TotalCharges"].replace({" ":"0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
df.info()  # Display a concise summary of the DataFrame (data types, non-null counts)

#Checking for Class Disstribution of target column
print(df["Churn"].value_counts())

# --- 4. Exploratory Data Analysis (EDA) ---
df.shape  # Check dimensions after cleaning
df.columns  # List all column names
df.head(22)  # Display the first 22 rows for a more detailed look
df.describe()  # Get descriptive statistics for numerical columns

# Function to plot a histogram for a given numerical column
def plot_histogram(df, column_name):
    plt.figure(figsize=(5,3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    #Calc mean/median
    col_mean = df[column_name].mean()  # Calculate the mean
    col_median = df[column_name].median()
    #add vertical lines
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")

    plt.legend()

    plt.show()

# Plot histograms for numerical features
plot_histogram(df, 'tenure')
plot_histogram(df,"MonthlyCharges")

# Function to plot a box plot to identify outliers
def plot_boxplot(df, column_name):
    plt.figure(figsize=(5,3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Box Plot of {column_name}")
    plt.ylabel(column_name)
    plt.show()

# Plot boxplots for numerical features
plot_boxplot(df,'tenure')
plot_boxplot(df, 'MonthlyCharges')
plot_boxplot(df, 'TotalCharges')

# Plot a heatmap to show correlations between numerical features
plt.figure(figsize=(8,4))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True, cmap="coolwarm",fmt='.2f')
plt.title('correlation map')
plt.show()

object_cols = df.select_dtypes(include= 'object').columns.to_list()  # Get list of categorical columns
object_cols = ['SeniorCitizen'] + object_cols  # Add 'SeniorCitizen' as it's a categorical feature (0 or 1)
object_cols

for col in object_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    plt.show()

# --- 5. Feature Engineering ---
df.head(5)  # Check the data before encoding

# Convert the target variable 'Churn' to numerical format (1 for "Yes", 0 for "No")
df["Churn"] = df["Churn"].replace({"Yes" : 1, "No" : 0})
df.head(5)  # Verify the change
df.info()  # Check data types after encoding 'Churn'
df["Churn"].value_counts()  # Check the distribution of the encoded 'Churn' column

#identifying columns with objext data type
object_columns = df.select_dtypes(include="object").columns

#initialize a dict to save the encoders
encoders={}

#apply label encoding and store the encoders
for column in object_columns:
    labelencoder = LabelEncoder()
    df[column] = labelencoder.fit_transform(df[column])
    encoders[column] = labelencoder


#save the encoder to a pickle file
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# --- 6. Model Preparation ---
# splitting the features and target
X = df.drop(columns=["Churn"])  # Features (all columns except 'Churn')
Y = df["Churn"]  # Target variable

#split training and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Handle class imbalance in the training data using SMOTE
smote = SMOTE(random_state = 42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

# --- 7. Model Training and Cross-Validation ---
models = {
    "Decision Tree" : DecisionTreeClassifier(random_state=42),
    "Random Forest" : RandomForestClassifier(random_state=42),
    "XGBoost" : XGBClassifier(random_state=42)
}

#dict to store cross validation results
cv_scores = {}

#perform 5 fold cross validation for each model
for model_name, model in models.items():
    print(f"Training {model_name} witth default parameters")
    scores = cross_val_score(model, X_train_smote, Y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")

# --- 8. Final Model Training and Evaluation ---
# Initialize and train the final model (Random Forest was chosen)
rfc = RandomForestClassifier(random_state=42)
rfc.fit (X_train_smote, Y_train_smote)

#evaluate on test data
Y_test_pred = rfc.predict(X_test)

# Print evaluation metrics
print("Accuracy Score: ", accuracy_score(Y_test, Y_test_pred))
print("Confusion Matrix: ", confusion_matrix(Y_test, Y_test_pred))
print("Classification Report: ", classification_report(Y_test, Y_test_pred))

# --- 9. Save the Model ---
#save trained model as pickle file
model_data = {"model" : rfc, "features_names" : X.columns.tolist()}  # Store model and feature names

with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# --- 10. Load and Verify the Saved Model ---
# Load the model from the pickle file
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]  # Extract the model
feature_names = model_data["features_names"]  # Extract the feature names
print(loaded_model)  # Print the loaded model object to verify
print(feature_names)  # Print the list of feature names to verify

# --- 11. Make Predictions on New Data ---
# Create a dictionary with new customer data for a single prediction
customer_data = {
    'gender': 'Female',  # Example value for gender
    'SeniorCitizen': 0,  # Example value for SeniorCitizen (0 for No, 1 for Yes)
    'Partner': 'Yes',  # Example value for Partner
    'Dependents': 'No',  # Example value for Dependents
    'tenure': 1,  # Example value for tenure in months
    'PhoneService': 'No',  # Example value for PhoneService
    'MultipleLines': 'No phone service',  # Example value for MultipleLines
    'InternetService': 'DSL',  # Example value for InternetService
    'OnlineSecurity': 'No',  # Example value for OnlineSecurity
    'OnlineBackup': 'Yes',  # Example value for OnlineBackup
    'DeviceProtection': 'No',  # Example value for DeviceProtection
    'TechSupport': 'No',  # Example value for TechSupport
    'StreamingTV': 'No',  # Example value for StreamingTV
    'StreamingMovies': 'No',  # Example value for StreamingMovies
    'Contract': 'Month-to-month',  # Example value for Contract
    'PaperlessBilling': 'Yes',  # Example value for PaperlessBilling
    'PaymentMethod': 'Electronic check',  # Example value for PaymentMethod
    'MonthlyCharges': 29.85,  # Example value for MonthlyCharges
    'TotalCharges': 29.85  # Example value for TotalCharges
}

# Convert the new customer data dictionary to a pandas DataFrame
input_data_df = pd.DataFrame([customer_data])

# Load the saved encoders from the pickle file
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

print("Input data before encoding:\n", input_data_df.head())  # Display the input data before transforming it

# Apply the loaded label encoders to the categorical columns of the input DataFrame
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

print("\nInput data after encoding:\n", input_data_df.head())  # Display the input data after it has been encoded

# Make a prediction using the loaded model
prediction = loaded_model.predict(input_data_df)
# Get the prediction probabilities for each class (No Churn, Churn)
prob_model = loaded_model.predict_proba(input_data_df) 
print("\nRaw prediction array:", prediction)  # Print the raw prediction (e.g., [0] or [1])

# Print the final prediction in a user-friendly format
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
# Print the probabilities for the prediction
print(f"Prediction Probability: {prob_model}")  # Shows probability for [No Churn, Churn]