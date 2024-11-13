# Diabetes Prediction Model

This repository contains a machine learning model that predicts the likelihood of diabetes based on clinical data. The model employs multiple classification algorithms, such as **Decision Tree** and **XGBoost**, and utilizes **hyperparameter tuning** and **cross-validation** to ensure robust predictions. The project also includes feature scaling to standardize the dataset for better performance.

## Project Overview:
The goal of this project is to predict whether a patient is at risk of developing diabetes based on their clinical data, including factors like age, BMI, insulin levels, and more. The prediction model is built using scikit-learn and XGBoost, with a focus on optimizing model performance through fine-tuning.

### Key Features:
- **Data Preprocessing**: Handles missing values, feature scaling, and outlier detection.
- **Model Selection**: Trained using multiple classification models, including Decision Tree and XGBoost.
- **Hyperparameter Tuning**: Uses GridSearchCV to find optimal hyperparameters for each model.
- **Model Evaluation**: Evaluates model performance using accuracy, confusion matrix, and classification report.
- **Model Persistence**: The final model is serialized using `joblib` for easy deployment and prediction.

### Project Structure:
- **diabetes.csv**: Raw dataset containing clinical information of patients.
- **diabetes_prediction_model_with_tuning.pkl**: Saved, trained model after hyperparameter tuning.
- **scaler.pkl**: Saved scaler used for feature scaling.
- **decode.py**: Script to load and decode the saved model and scaler.
- **model_comparison.py**: Script that compares the performance of various models.

### Dependencies:
- `scikit-learn`
- `xgboost`
- `pandas`
- `numpy`
- `joblib`
- `matplotlib`
- `seaborn`

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-model.git](https://github.com/kajalkoirala/Diabetes-Prediction-Model)
   ```

2. **Load the trained model**:

   To make predictions using the trained model, you can load it with the following code:

   ```python
   import joblib

   # Load the trained model
   model = joblib.load('diabetes_prediction_model_with_tuning.pkl')

   # Load the scaler (used for feature scaling)
   scaler = joblib.load('scaler.pkl')

   # Sample input data (replace with real data)
   input_data = [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]  # Example input
   scaled_data = scaler.transform(input_data)

   # Predict diabetes risk (0: No, 1: Yes)
   prediction = model.predict(scaled_data)
   print("Predicted Outcome:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
   ```

3. **Train the model (if needed)**:
   You can train the model from scratch using the `model_comparison.py` script:
   ```bash
   python model_comparison.py
   ```
   This will train multiple models and save the best one as `diabetes_prediction_model_with_tuning.pkl`.

## Model Evaluation
The models are evaluated using multiple metrics:
- **Accuracy**: Measures the overall correctness of the model.
- **Confusion Matrix**: A matrix that shows the true vs. predicted values.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

## Contributing
Contributions are welcome! Feel free to fork the repository, create a branch, make your changes, and submit a pull request. Please make sure to update the documentation and write tests for your changes.

