import numpy as np
from sklearn.datasets import make_circles, make_gaussian_quantiles, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

def load_circle_dataset(random_state=42):
    radius = np.sqrt(2 / np.pi)
    X, y = make_circles(
        n_samples=1000,
        noise=0.1,
        factor=0.7,
        random_state=random_state
    )
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = 1 - y  # 圆内为1，圆外为0
    return X, y

def load_gaussian_mixture_dataset(random_state=42):
    """生成论文指定的高斯混合数据集"""
    X1, y1 = make_gaussian_quantiles(mean=[2]*10, cov=1.0,
                                    n_samples=333, n_features=10, random_state=random_state)
    X2, y2 = make_gaussian_quantiles(mean=[-2]*10, cov=1.0,
                                    n_samples=333, n_features=10, random_state=random_state+1)
    X3, y3 = make_gaussian_quantiles(mean=[0]*10, cov=1.0,
                                    n_samples=334, n_features=10, random_state=random_state+2)
    
    X = np.vstack((X1, X2, X3))
    y = np.hstack((y1, y2 + 1, y3 + 2))
    y = np.where(y == 0, 0, 1)  # Mode 1→0, Modes 2-3→1
    return X, y

def load_breast_cancer_dataset(random_state=42):
    """加载乳腺癌数据集"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    X, y = shuffle(X, y, random_state=random_state)
    y = 1 - y  # 良性→0，恶性→1
    
    # 高维数据降维处理
    pca = PCA(n_components=0.95, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced, y

def preprocess_dataset(X_train, X_val, X_test):
    """预处理数据集：标准化到[0, 1]范围"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def prepare_datasets(dataset_name, random_state=42):
    if dataset_name == "circle":
        X, y = load_circle_dataset(random_state=random_state)
        train_size, test_size = 700, 300
    elif dataset_name == "gaussian":
        X, y = load_gaussian_mixture_dataset(random_state=random_state)
        train_size, test_size = 700, 300
    elif dataset_name == "breast_cancer":
        X, y = load_breast_cancer_dataset(random_state=random_state)
        train_size, test_size = 400, 169
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 划分训练集和测试集
    X_train, X_test = X[:train_size], X[train_size:train_size+test_size]
    y_train, y_test = y[:train_size], y[train_size:train_size+test_size]
    
    # 划分验证集
    val_size = int(train_size * 0.2)
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    
    # 预处理
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_dataset(
        X_train, X_val, X_test
    )
    
    # 打印数据集信息
    print(f"数据集: {dataset_name}")
    print(f"训练集: {X_train_scaled.shape}, 验证集: {X_val_scaled.shape}, 测试集: {X_test_scaled.shape}")
    print(f"特征维度: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

if __name__ == "__main__":
    for dataset in ["circle", "gaussian", "breast_cancer"]:
        print("\n" + "="*60)
        prepare_datasets(dataset_name=dataset)
