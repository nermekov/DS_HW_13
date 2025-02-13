import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Assuming the dataset has columns 'YearsExperience' and 'Salary'
    X = data[['YearsExperience']]
    y = data['Salary']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test