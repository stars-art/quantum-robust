from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import dask
import time
from dask.distributed import LocalCluster, Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from .data import circle, gaussian, breast_cancer
from .file import save_training_nqedr_output
from .circuit import nqedr_encoding, lipschitz_regularization, parameter_regularization, calculate_accuracy
from .data_types import NQEDRTrainingOutput
import logging

# 实验配置
n_training_samples = 1000
n_validation_samples = 200
n_layer = 3
n_qubit = 3
n_ensemble = 5
learning_rate = 0.01
n_epoch = 50
early_stopping_patience = 15
scaling_range = (0, 2 * np.pi)
cv_folds = 5  # 5折交叉验证

# 双正则化参数搜索范围
lambda1_candidates = [0.0, 0.001, 0.01, 0.05, 0.1]  # 参数正则化
lambda2_candidates = [0.0, 0.1, 0.2, 0.3, 0.5]  # 梯度正则化

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NQE-DR-Trainer")


def preprocess_data(x_train, x_validation):
    """数据标准化处理"""
    scaler = MinMaxScaler(feature_range=scaling_range)
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    return x_train_scaled, x_validation_scaled, scaler


def train_nqedr(lamb1, lamb2, weights, biases, x_train, y_train, x_validation, y_validation):
    """NQE-DR编码方式的训练函数，带双正则化"""
    opt = AdamOptimizer(learning_rate)
    best_val_accuracy = 0
    best_weights = weights.copy()
    best_biases = biases.copy()
    patience_counter = 0

    # 存储训练历史
    cost_history = []
    train_acc_history = []
    val_acc_history = []
    param_reg_history = []
    lipschitz_history = []

    for epoch in range(n_epoch):
        # 带双正则化的成本函数
        def cost(weights, biases):
            # 基础损失
            predictions = np.array([np.sign(nqedr_encoding(weights, biases, x)) for x in x_train])
            loss = np.mean((predictions - y_train) ** 2)

            # 参数正则化 (λ1)
            param_reg = parameter_regularization(weights, biases)


            lipschitz_reg = lipschitz_regularization(weights)

            # 总损失
            return loss + lamb1 * param_reg + lamb2 * lipschitz_reg, param_reg, lipschitz_reg

        # 优化步骤
        (current_cost, param_reg, lipschitz_reg), _, _ = opt.step_and_cost(cost, weights, biases)

        # 计算准确率
        train_predictions = np.array([np.sign(nqedr_encoding(weights, biases, x)) for x in x_train])
        train_accuracy = calculate_accuracy(y_train, train_predictions)

        val_predictions = np.array([np.sign(nqedr_encoding(weights, biases, x)) for x in x_validation])
        val_accuracy = calculate_accuracy(y_validation, val_predictions)

        # 记录历史
        cost_history.append(current_cost)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)
        param_reg_history.append(param_reg)
        lipschitz_history.append(lipschitz_reg)

        # 早停机制
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = weights.copy()
            best_biases = biases.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (λ1={lamb1}, λ2={lamb2})")
                break

        # 打印进度
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Cost={current_cost:.4f}, Train Acc={train_accuracy:.4f}, "
                        f"Val Acc={val_accuracy:.4f}, Param Reg={param_reg:.4f}, Lip Reg={lipschitz_reg:.4f}")

    return (best_val_accuracy, best_weights, best_biases, np.array(cost_history),
            np.array(train_acc_history), np.array(val_acc_history),
            np.array(param_reg_history), np.array(lipschitz_history))


def select_best_regularizers(x, y):
    """通过5折交叉验证选择最佳λ1和λ2参数"""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = np.zeros((len(lambda1_candidates), len(lambda2_candidates)))
    x_dim = x.shape[1]

    # 遍历所有参数组合
    for i, lamb1 in enumerate(lambda1_candidates):
        for j, lamb2 in enumerate(lambda2_candidates):
            fold_scores = []

            for train_idx, val_idx in kf.split(x):
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 初始化参数
                weights = np.random.normal(loc=0.0, scale=0.1,
                                           size=(n_layer, n_qubit, x_dim, x_dim),
                                           requires_grad=True)
                biases = np.random.normal(loc=0.0, scale=0.1,
                                          size=(n_layer, n_qubit, x_dim),
                                          requires_grad=True)

                # 训练模型（简化版）
                val_acc, _, _, _, _, _, _, _ = train_nqedr(
                    lamb1, lamb2, weights, biases, x_train, y_train, x_val, y_val
                )
                fold_scores.append(val_acc)

            # 计算平均分数
            cv_scores[i, j] = np.mean(fold_scores)
            logger.info(f"CV: λ1={lamb1}, λ2={lamb2} - 平均准确率: {cv_scores[i, j]:.4f}")

    # 找到最佳参数
    best_idx = np.unravel_index(np.argmax(cv_scores), cv_scores.shape)
    best_lamb1 = lambda1_candidates[best_idx[0]]
    best_lamb2 = lambda2_candidates[best_idx[1]]

    logger.info(f"NQE-DR最佳正则化参数: λ1={best_lamb1}, λ2={best_lamb2}")
    return best_lamb1, best_lamb2


def run(dataset_name="circle"):
    """运行NQE-DR编码模型训练"""
    # 初始化Dask集群
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB')
    client = Client(cluster)
    logger.info(f"Dask Dashboard: {client.dashboard_link}")

    # 加载数据集
    if dataset_name == "circle":
        x_train, y_train = circle(n_training_samples)
        x_val, y_val = circle(n_validation_samples)
    elif dataset_name == "gaussian":
        x_train, y_train = gaussian(n_training_samples)
        x_val, y_val = gaussian(n_validation_samples)
    elif dataset_name == "breast_cancer":
        x_train, y_train = breast_cancer(n_training_samples)
        x_val, y_val = breast_cancer(n_validation_samples)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 数据预处理
    x_train_scaled, x_val_scaled, scaler = preprocess_data(x_train, x_val)
    x_dim = x_train_scaled.shape[1]

    # 交叉验证选择最佳正则化参数
    best_lamb1, best_lamb2 = select_best_regularizers(x_train_scaled, y_train)

    # 初始化集成模型参数
    weights = np.random.normal(loc=0.0, scale=0.1,
                               size=(n_ensemble, n_layer, n_qubit, x_dim, x_dim),
                               requires_grad=True)
    biases = np.random.normal(loc=0.0, scale=0.1,
                              size=(n_ensemble, n_layer, n_qubit, x_dim),
                              requires_grad=True)

    # 并行训练集成模型
    start_time = time.time()
    jobs = [dask.delayed(train_nqedr)(
        best_lamb1, best_lamb2, weights[i], biases[i],
        x_train_scaled, y_train, x_val_scaled, y_val
    ) for i in range(n_ensemble)]

    results = dask.compute(*jobs)
    end_time = time.time()
    logger.info(f"总训练时间: {round((end_time - start_time) / 60, 2)}分钟")

    # 整理结果
    max_epochs = n_epoch
    cost_over_epochs = np.zeros((n_ensemble, max_epochs))
    train_accuracy_over_epochs = np.zeros((n_ensemble, max_epochs))
    validation_accuracy_over_epochs = np.zeros((n_ensemble, max_epochs))
    weights_over_epochs = np.zeros((n_ensemble, n_layer, n_qubit, x_dim, x_dim))
    biases_over_epochs = np.zeros((n_ensemble, n_layer, n_qubit, x_dim))
    param_reg_over_epochs = np.zeros((n_ensemble, max_epochs))
    lipschitz_reg_over_epochs = np.zeros((n_ensemble, max_epochs))
    best_validation_accuracies = []

    for i in range(n_ensemble):
        (best_val_acc, best_weights, best_biases, cost_hist,
         train_acc_hist, val_acc_hist, param_reg_hist, lipschitz_hist) = results[i]

        best_validation_accuracies.append(best_val_acc)
        weights_over_epochs[i] = best_weights
        biases_over_epochs[i] = best_biases

        actual_epochs = len(cost_hist)
        cost_over_epochs[i, :actual_epochs] = cost_hist
        train_accuracy_over_epochs[i, :actual_epochs] = train_acc_hist
        validation_accuracy_over_epochs[i, :actual_epochs] = val_acc_hist
        param_reg_over_epochs[i, :actual_epochs] = param_reg_hist
        lipschitz_reg_over_epochs[i, :actual_epochs] = lipschitz_hist

    # 保存结果
    output = NQEDRTrainingOutput(
        x_train=x_train, y_train=y_train,
        x_validation=x_val, y_validation=y_val,
        best_lambda1=best_lamb1,
        best_lambda2=best_lamb2,
        cost_over_epochs=cost_over_epochs,
        train_accuracy_over_epochs=train_accuracy_over_epochs,
        validation_accuracy_over_epochs=validation_accuracy_over_epochs,
        weights_over_epochs=weights_over_epochs,
        biases_over_epochs=biases_over_epochs,
        param_regularization_over_epochs=param_reg_over_epochs,
        lipschitz_regularization_over_epochs=lipschitz_reg_over_epochs
    )
    save_training_nqedr_output(output)

    # 打印总结
    logger.info("\n===== 训练总结 =====")
    logger.info(f"平均验证准确率: {np.mean(best_validation_accuracies):.4f} ± {np.std(best_validation_accuracies):.4f}")
    logger.info(f"最佳正则化参数: λ1={best_lamb1}, λ2={best_lamb2}")

    return output


if __name__ == "__main__":
    run(dataset_name="circle")
