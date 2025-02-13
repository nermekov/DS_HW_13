from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_and_preprocess_data
import joblib

def evaluate_model(data_path, model_path):
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    # Load the trained model
    model = joblib.load(model_path)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    data_path = "Salary_Data.csv"  # Update with the correct path if necessary
    model_path = "linear_regression_model.pkl"