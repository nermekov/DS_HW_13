import joblib
from sklearn.linear_model import LinearRegression
from preprocess import load_and_preprocess_data

def train_and_save_model(data_path, model_path):
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(model, model_path)
    
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    print("Training the model...")
    data_path = "Salary_Data.csv"  # Update with the correct path if necessary
    model_path = "linear_regression_model.pkl"
    train_and_save_model(data_path, model_path)