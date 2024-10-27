import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils import save_predictions_to_csv

def select_important_features(X_train, y_train, n_features=10):
    """
    Select the most important features using RandomForest feature importances.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feature_indices = importances.argsort()[-n_features:]
    selected_features = X_train.columns[feature_indices]
    return X_train[selected_features], selected_features

def build_mlp_model(input_dim):
    """
    Build a simple MLP model using Keras.
    
    Parameters:
    input_dim : int
        The dimension of the input features.
    
    Returns:
    model : Keras Sequential model
        The compiled MLP model.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    return model

dataset_names = []
X_trains = []
y_trains = []
X_tests = []

for folder_name in os.listdir("./Competition_data"):
    dataset_names.append(folder_name)
    X_trains.append(pd.read_csv(f"./Competition_data/{folder_name}/X_train.csv", header=0))
    y_trains.append(pd.read_csv(f"./Competition_data/{folder_name}/y_train.csv", header=0))
    X_tests.append(pd.read_csv(f"./Competition_data/{folder_name}/X_test.csv", header=0))

for i in range(len(dataset_names)):
    print(f"Processing dataset: {dataset_names[i]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_trains[i])
    X_test_scaled = scaler.transform(X_tests[i])
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_trains[i].columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_tests[i].columns)
    
    X_train_selected, selected_features = select_important_features(X_train_df, y_trains[i].values.ravel(), n_features=10)
    X_test_selected = X_test_df[selected_features]
    
    model_mlp = build_mlp_model(input_dim=X_train_selected.shape[1])
    
    model_mlp.fit(X_train_selected, y_trains[i].values.ravel(), epochs=20, batch_size=32, verbose=2, validation_split=0.2)
    
    y_prob_train = model_mlp.predict(X_train_selected).flatten()
    auc_score = roc_auc_score(y_trains[i], y_prob_train)
    print(f"Dataset: {dataset_names[i]} - MLP AUC Score: {auc_score:.4f}")
    
    y_prob_test = model_mlp.predict(X_test_selected).flatten()
    
    save_predictions_to_csv(y_prob_test, dataset_names[i], folder_path='./Competition_data/')
