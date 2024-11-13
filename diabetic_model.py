import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#------ Data preprocessing -----------------------

# Load the data
data = pd.read_csv('C://Users//kajal//OneDrive//Desktop//diabetic prediction//diabetes.csv')

# Impute missing values with the mean
data.fillna(data.mean(), inplace=True)

# Initial boxplot to show outliers
plt.figure(figsize=(20, 20))
sbn.boxplot(data=data, medianprops={'color': 'red', 'linewidth': 2})
plt.xticks(rotation=90)
plt.title("Boxplot Before Handling Outliers")
plt.show()

# Identify numerical columns (excluding 'Outcome' column)
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Function to remove outliers using the IQR method
def remove_outliers_iqr(df, numerical_columns):
    for col in numerical_columns:
        # Calculate Q1, Q3, and IQR for each numerical feature
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds to filter out the outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

# Apply outlier removal
data_no_outliers = remove_outliers_iqr(data, numerical_cols)

# Check the shape after removing outliers
print(f"Original data shape: {data.shape}")
print(f"Data shape after removing outliers: {data_no_outliers.shape}")

# Boxplot after handling outliers
plt.figure(figsize=(20, 20))
sbn.boxplot(data=data_no_outliers, medianprops={'color': 'red', 'linewidth': 2})
plt.xticks(rotation=90)
plt.title("Boxplot After Handling Outliers")
plt.show()

# Feature scaling
X = data_no_outliers.drop(columns=['Outcome'])  # Outcome is the target
y = data_no_outliers['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

#--------- Data preparation ----------------------

# Split data (70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=40)

#------ Model selection and training --------------

# Decision Tree
tree_model = DecisionTreeClassifier()

# XGBoost
xgboost_model = XGBClassifier(verbosity=1)

# Hyperparameter tuning for Decision Tree using GridSearchCV
param_grid_tree = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search_tree = GridSearchCV(estimator=tree_model, param_grid=param_grid_tree, cv=5, verbose=1, n_jobs=-1)
grid_search_tree.fit(x_train, y_train)
tree_model = grid_search_tree.best_estimator_

# Hyperparameter tuning for XGBoost using GridSearchCV
param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7]
}
grid_search_xgb = GridSearchCV(estimator=xgboost_model, param_grid=param_grid_xgb, cv=5, verbose=1, n_jobs=-1)
grid_search_xgb.fit(x_train, y_train)
xgboost_model = grid_search_xgb.best_estimator_

#------ Cross-validation for model evaluation------

# Perform cross-validation
cv_tree_scores = cross_val_score(tree_model, X_scaled_df, y, cv=5)
cv_xgb_scores = cross_val_score(xgboost_model, X_scaled_df, y, cv=5)

print(f"Decision Tree Cross-validation mean: {cv_tree_scores.mean()}")
print(f"XGBoost Cross-validation mean: {cv_xgb_scores.mean()}")

#------ Final Model Evaluation ------

# Evaluate on the test set
y_pred_tree = tree_model.predict(x_test)
y_pred_xgb = xgboost_model.predict(x_test)

# Accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

# Classification Reports
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrices
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Plot Confusion Matrices
plt.figure(figsize=(6, 6))
sbn.heatmap(cm_tree, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'], yticklabels=['Not Diabetic', 'Diabetic'])
plt.title("Decision Tree Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

plt.figure(figsize=(6, 6))
sbn.heatmap(cm_xgb, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'], yticklabels=['Not Diabetic', 'Diabetic'])
plt.title("XGBoost Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Model comparison
print(f"Decision Tree Accuracy: {accuracy_tree:.4f}")
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")

# Model Performance Comparison Visualization
model_results = {
    'Decision Tree': accuracy_tree,
    'XGBoost': accuracy_xgb
}
performance_df = pd.DataFrame(list(model_results.items()), columns=['Model', 'Accuracy'])

plt.figure(figsize=(10, 6))
sbn.barplot(x='Model', y='Accuracy', data=performance_df, palette='viridis')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.show()

# Save the best model (e.g., XGBoost)
joblib.dump(xgboost_model, 'diabetes_prediction_model_with_tuning.pkl')

# Saving the scaler
joblib.dump(scaler, 'scaler.pkl')

# Load saved model and scaler
loaded_model = joblib.load('diabetes_prediction_model_with_tuning.pkl')
scaler = joblib.load('scaler.pkl')

# Function to ask user input for prediction
def get_user_input():
    # Prompt the user for each feature (make sure to provide a description of each feature)
    age = float(input("Enter Age: "))
    glucose = float(input("Enter Glucose level: "))
    blood_pressure = float(input("Enter Blood Pressure: "))
    skin_thickness = float(input("Enter Skin Thickness: "))
    insulin = float(input("Enter Insulin: "))
    bmi = float(input("Enter BMI: "))
    diabetes_pedigree = float(input("Enter Diabetes Pedigree Function: "))
    pregnancy = float(input("Enter Number of Pregnancies: "))
    
    # Return the inputs as a list (same order as in the model's training data)
    return [pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

# Get the user's input
user_input = get_user_input()

# Scale the user input (same way as training data)
scaled_input = scaler.transform([user_input])

# Predict whether the user is diabetic using the loaded model
prediction = loaded_model.predict(scaled_input)

# Display the result
if prediction == 0:
    print("The prediction is: Not Diabetic")
else:
    print("The prediction is: Diabetic")
