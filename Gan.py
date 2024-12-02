import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# GAN 的參數
latent_dim = 10
epochs = 1000
batch_size = 64
lr = 0.0002

# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim)
        )

    def forward(self, z):
        return self.model(z)

# 判別器
class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 數據預處理函數
def preprocess_data(X_train, y_train, X_test):
    # 分離數值型和類別型特徵
    numeric_features = X_train.select_dtypes(include=['float']).columns
    categorical_features = X_train.select_dtypes(include=['int']).columns

    # 標準化數值型特徵
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train[numeric_features])
    X_test_numeric = scaler.transform(X_test[numeric_features])

    # 編碼類別型特徵
    if len(categorical_features) > 0:
        encoder = LabelEncoder()
        for col in categorical_features:
            X_train[col] = encoder.fit_transform(X_train[col])
            X_test[col] = encoder.transform(X_test[col])

    # 合併數值型和類別型特徵
    X_train_processed = np.hstack((X_train_numeric, X_train[categorical_features].values))
    X_test_processed = np.hstack((X_test_numeric, X_test[categorical_features].values))

    return X_train_processed, y_train, X_test_processed

def train_gan(minority_data, latent_dim, data_dim):
    generator = Generator(latent_dim, data_dim)
    discriminator = Discriminator(data_dim)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    minority_data = torch.tensor(minority_data, dtype=torch.float32)

    for epoch in range(epochs):
        d_loss, g_loss = 0, 0  # 初始化損失值

        # 如果樣本數量不足一個批次，將全部數據作為一個批次
        if len(minority_data) < batch_size:
            batch_size_actual = len(minority_data)
            real_batch = minority_data
            real_labels = torch.ones((batch_size_actual, 1))

            # 生成假數據
            z = torch.randn((batch_size_actual, latent_dim))
            fake_batch = generator(z)
            fake_labels = torch.zeros((batch_size_actual, 1))

            # 訓練判別器
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_batch), real_labels)
            fake_loss = criterion(discriminator(fake_batch.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # 訓練生成器
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_batch), real_labels)
            g_loss.backward()
            optimizer_G.step()

        else:  # 正常批次訓練
            for _ in range(len(minority_data) // batch_size):
                # 真實數據
                real_batch = minority_data[np.random.randint(0, len(minority_data), batch_size)]
                real_labels = torch.ones((batch_size, 1))

                # 假數據
                z = torch.randn((batch_size, latent_dim))
                fake_batch = generator(z)
                fake_labels = torch.zeros((batch_size, 1))

                # 訓練判別器
                optimizer_D.zero_grad()
                real_loss = criterion(discriminator(real_batch), real_labels)
                fake_loss = criterion(discriminator(fake_batch.detach()), fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()

                # 訓練生成器
                optimizer_G.zero_grad()
                g_loss = criterion(discriminator(fake_batch), real_labels)
                g_loss.backward()
                optimizer_G.step()

        # 打印損失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # 生成新數據
    z = torch.randn((500, latent_dim))
    synthetic_data = generator(z).detach().numpy()
    return synthetic_data

def process_all_datasets(data_dir, output_dir):
    dataset_names = []
    X_trains, y_trains, X_tests = [], [], []
    skipped_datasets = []  # 用於記錄跳過的數據集名稱

    # 創建輸出目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍歷所有數據集目錄
    for folder_name in sorted(os.listdir(data_dir)):
        dataset_names.append(folder_name)
        X_trains.append(pd.read_csv(f"{data_dir}/{folder_name}/X_train.csv"))
        y_trains.append(pd.read_csv(f"{data_dir}/{folder_name}/y_train.csv").values.ravel())
        X_tests.append(pd.read_csv(f"{data_dir}/{folder_name}/X_test.csv"))

    # 處理每個數據集
    for i, dataset_name in enumerate(dataset_names):
        print(f"Processing dataset: {dataset_name}")
        try:
            X_train, y_train, X_test = preprocess_data(X_trains[i], y_trains[i], X_tests[i])

            # 檢查 y_train 是否為整數型
            if not np.issubdtype(y_train.dtype, np.integer):
                if np.all(np.mod(y_train, 1) == 0):  # 標籤是浮點型但實際為整數
                    y_train = y_train.astype(int)
                else:
                    raise ValueError(f"y_train contains non-integer or invalid values: {y_train}")

            # 檢查類別不平衡
            class_counts = np.bincount(y_train)
            minority_class = np.argmin(class_counts)
            majority_class = np.argmax(class_counts)

            if class_counts[minority_class] < class_counts[majority_class]:
                print(f"Class distribution: {class_counts}")
                print(f"Minority class: {minority_class}, Majority class: {majority_class}")

                # 提取少數類別數據
                minority_data = X_train[y_train == minority_class]

                # 使用 GAN 生成少數類別數據
                synthetic_data = train_gan(minority_data, latent_dim, X_train.shape[1])
                synthetic_labels = np.full(len(synthetic_data), minority_class)

                # 合併新舊數據
                X_augmented = np.vstack((X_train, synthetic_data))
                y_augmented = np.hstack((y_train, synthetic_labels))

                print(f"Augmented dataset shape: {X_augmented.shape}, {y_augmented.shape}")

                # 創建對應的輸出資料夾結構
                dataset_output_dir = os.path.join(output_dir, dataset_name)
                if not os.path.exists(dataset_output_dir):
                    os.makedirs(dataset_output_dir)

                # 獲取原始特徵名，為生成文件添加標頭
                feature_columns = X_trains[i].columns.tolist()
                target_column = 'target'

                # 分開保存特徵和標籤
                pd.DataFrame(X_augmented, columns=feature_columns).to_csv(
                    f"{dataset_output_dir}/X_train_augmented.csv", index=False
                )
                pd.DataFrame(y_augmented, columns=[target_column]).to_csv(
                    f"{dataset_output_dir}/y_train_augmented.csv", index=False
                )
                print(f"Augmented features saved: {dataset_output_dir}/X_train_augmented.csv")
                print(f"Augmented labels saved: {dataset_output_dir}/y_train_augmented.csv\n")

        except Exception as e:
            print(f"Skipping dataset {dataset_name} due to error: {e}")
            skipped_datasets.append(dataset_name)

    # 輸出跳過的數據集
    print("\nSummary:")
    print(f"Total datasets processed: {len(dataset_names) - len(skipped_datasets)}")
    print(f"Total datasets skipped: {len(skipped_datasets)}")
    if skipped_datasets:
        print("Skipped datasets:")
        for skipped in skipped_datasets:
            print(f" - {skipped}")

# 執行主函數
process_all_datasets(data_dir="./Competition_data", output_dir="./Augmented_data")