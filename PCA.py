from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_trains[i])
X_test_scaled = scaler.transform(X_tests[i])

X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, n_components=0.95)
