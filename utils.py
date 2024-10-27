from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

def calculate_auc_score(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    return auc_score

def compare_auc_scores(model1_auc_scores, model2_auc_scores, dataset_names):
    differences = {}
    for i in range(len(dataset_names)):
        diff = model1_auc_scores[i] - model2_auc_scores[i]
        differences[dataset_names[i]] = diff
        print(f"Dataset: {dataset_names[i]} - AUC Difference: {diff:.4f} (Model 1 - Model 2)")
    return differences

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def save_predictions_to_csv(predictions, dataset_name, folder_path='./Competition_data/'):
    """
    Save binary predictions to CSV file.
    Convert predicted probabilities to 0 or 1 based on a threshold of 0.5.
    """
    # Convert probabilities to binary predictions (0 or 1)
    binary_predictions = (predictions >= 0.5).astype(int)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(binary_predictions, columns=['y_predict_proba'])
    df.to_csv(f'{folder_path}{dataset_name}/y_predict.csv', index=False, header=True)
