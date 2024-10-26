from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def train_knn(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train.squeeze())
    return model

def train_logistic_regression(X_train, y_train, max_iter=1000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train.squeeze())
    return model

def train_svm(X_train, y_train, kernel='poly', degree=2):
    model = SVC(kernel=kernel, degree=degree, probability=True)
    model.fit(X_train, y_train.squeeze())
    return model

def train_neural_network(X_train, y_train, hidden_layers=(100,), max_iter=500):
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=max_iter)
    model.fit(X_train, y_train.squeeze())
    return model

def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train.squeeze())
    return model