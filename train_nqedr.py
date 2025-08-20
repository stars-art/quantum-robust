from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import dask
import time
from dask.distributed import LocalCluster, Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from .data import circle, gaussian, breast_cancer
from .file import save_training_nqedr_output
from .circuit import nqedr_encoding, calculate_accuracy
from .data_types import NQEDRTrainingOutput
import logging

# 实验配置
n_training_samples = 1000
n_validation_samples = 200
n_layer = 3
n_qubit = 3
n_ensemble = 5
learning_rate = 0.01
n_epoch = 200
early_stopping_patience = 15
scaling_range = (0, 2 * np.pi)
cv_folds = 5  # 5折交叉验证


lambda1_candidates = [0.001, 0.01, 0.05, 0.1, 0.2]  # 参数正则化强度 λ1
lambda2_candidates = [0.1, 0.2, 0.3, 0.5, 0.7]      # 梯度正则化强度 λ2

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NQE-DR-Trainer")


def preprocess_data(x_train, x_validation):
    """数据标准化处理
    scaler = MinMaxScaler(feature_range=scaling_range)
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    return x_train_scaled, x_validation_scaled, scaler


def parameter_regularization(theta):
    """参数正则化 R1 = λ1 · ∥Θ∥²_F（Frobenius范数）"""
    # Θ 包含所有可训练参数（权重和偏置）
    return np.sum([np.sum(param**2) for param in theta])


def gradient_regularization(weights, biases, x):
    """梯度正则化 R2 = λ2 · ∥∇xNQE-DR(x)∥²（L2范数）"""
    # 计算模型输出对输入的梯度
    gradients = []
    for i in range(x.shape[1]):
        # 对每个输入特征求偏导
        def partial_deriv(x_i):
            x_perturbed = x.copy()
            x_perturbed[i] = x_i
            return nqedr_encoding(weights, biases, x_perturbed)
        
        # 使用数值微分计算梯度（量子模型通常无法直接求导）
        grad = np.gradient(partial_deriv(x[i]))[0]
        gradients.append(grad**2)
    
    return np.sum(gradients)


def train_nqedr(lamb1, lamb2, weights, biases, x_train, y_train, x_validation, y_validation):
    """NQE-DR编码方式的训练函数，"""
    opt = AdamOptimizer(learning_rate)
    best_val_accuracy = 0
    best_weights = weights.copy()
    best_biases = biases.copy()
    patience_counter = 0

    # 存储训练历史
    cost_history = []
    train_acc_history = []
    val_acc_history = []
    param_reg_history = []  # R1 = λ1·∥Θ∥²_F
    gradient_reg_history = []  # R2 = λ2·∥∇xNQE-DR∥²
    lipschitz_bound_history = []  #  Lipschitz界 L = sup∥∇xNQE-DR∥

    for epoch in range(n_epoch):
        # 定义完整损失函数 Ltotal = Ltask + R1 + R2
        def total_loss(weights, biases):
            # 1. 任务损失 Ltask（分类任务使用交叉熵损失）
            predictions = np.array([nqedr_encoding(weights, biases, x) for x in x_train])
            # 交叉熵损失（适用于分类任务）
            epsilon = 1e-8  # 防止log(0)
            y_probs = (predictions + 1) / 2  # 将[-1,1]转换为[0,1]概率
            loss_task = -np.mean(y_train * np.log(y_probs + epsilon) + 
                                (1 - y_train) * np.log(1 - y_probs + epsilon))

            # 2. 参数正则化 R1 = λ1·∥Θ∥²_F
            theta = [weights, biases]  # 所有可训练参数 Θ
            r1 = lamb1 * parameter_regularization(theta)

            # 3. 梯度正则化 R2 = λ2·∥∇xNQE-DR∥²
            # 对训练集样本的梯度正则化取平均
            r2 = lamb2 * np.mean([gradient_regularization(weights, biases, x) for x in x_train])

            # 总损失
            total = loss_task + r1 + r2
            
            # 计算Lipschitz界 L = sup∥∇xNQE-DR∥
            gradients = [np.sqrt(gradient_regularization(weights, biases, x)) for x in x_train]
            lipschitz_bound = np.max(gradients) if gradients else 0.0

            return total, loss_task, r1, r2, lipschitz_bound

        # 优化步骤
        (current_loss, loss_task, r1, r2, lipschitz_bound), _, _ = opt.step_and_cost(
            total_loss, weights, biases
        )

        # 计算准确率
        train_predictions = np.array([np.sign(nqedr_encoding(weights, biases, x)) for x in x_train])
        train_accuracy = calculate_accuracy(y_train, train_predictions)

        val_predictions = np.array([np.sign(nqedr_encoding(weights, biases, x)) for x in x_validation])
        val_accuracy = calculate_accuracy(y_validation, val_predictions)

        # 记录历史
        cost_history.append(current_loss)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)
        param_reg_history.append(r1)  # 记录R1 = λ1·∥Θ∥²_F
        gradient_reg_history.append(r2)  # 记录R2 = λ2·∥∇xNQE-DR∥²
        lipschitz_bound_history.append(lipschitz_bound)  # 记录Lipschitz界

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

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Ltotal={current_loss:.4f}, Ltask={loss_task:.4f}, "
                f"R1={r1:.4f}, R2={r2:.4f}, L={lipschitz_bound:.4f}, "
                f"Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}"
            )

    return (best_val_accuracy, best_weights, best_biases, np.array(cost_history),
            np.array(train_acc_history), np.array(val_acc_history),
            np.array(param_reg_history), np.array(gradient_reg_history),
            np.array(lipschitz_bound_history))


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

                # 初始化参数 Θ（权重和偏置）
                weights = np.random.normal(loc=0.0, scale=0.1,
                                           size=(n_layer, n_qubit, x_dim, x_dim),
                                           requires_grad=True)
                biases = np.random.normal(loc=0.0, scale=0.1,
                                          size=(n_layer, n_qubit, x_dim),
                                          requires_grad=True)

                # 训练模型
                val_acc, _, _, _, _, _, _, _, _ = train_nqedr(
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

    # 确保标签为0/1（适应交叉熵损失）
    y_train = (y_train + 1) / 2 if np.min(y_train) == -1 else y_train
    y_val = (y_val + 1) / 2 if np.min(y_val) == -1 else y_val

    # 数据预处理
    x_train_scaled, x_val_scaled, scaler = preprocess_data(x_train, x_val)
    x_dim = x_train_scaled.shape[1]

    # 交叉验证选择最佳正则化参数
    best_lamb1, best_lamb2 = select_best_regularizers(x_train_scaled, y_train)

    # 初始化集成模型参数 Θ
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
    param_reg_over_epochs = np.zeros((n_ensemble, max_epochs))  # R1历史
    lipschitz_reg_over_epochs = np.zeros((n_ensemble, max_epochs))  # R2历史
    lipschitz_bound_history = np.zeros((n_ensemble, max_epochs))  # Lipschitz界历史
    best_validation_accuracies = []

    for i in range(n_ensemble):
        (best_val_acc, best_weights, best_biases, cost_hist,
         train_acc_hist, val_acc_hist, param_reg_hist, grad_reg_hist,
         lip_bound_hist) = results[i]

        best_validation_accuracies.append(best_val_acc)
        weights_over_epochs[i] = best_weights
        biases_over_epochs[i] = best_biases

        actual_epochs = len(cost_hist)
        cost_over_epochs[i, :actual_epochs] = cost_hist
        train_accuracy_over_epochs[i, :actual_epochs] = train_acc_hist
        validation_accuracy_over_epochs[i, :actual_epochs] = val_acc_hist
        param_reg_over_epochs[i, :actual_epochs] = param_reg_hist  # 存储R1
        lipschitz_reg_over_epochs[i, :actual_epochs] = grad_reg_hist  # 存储R2
        lipschitz_bound_history[i, :actual_epochs] = lip_bound_hist  # 存储Lipschitz界

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
        param_regularization_over_epochs=param_reg_over_epochs,  # R1 = λ1·∥Θ∥²_F
        lipschitz_regularization_over_epochs=lipschitz_reg_over_epochs  # R2 = λ2·∥∇xNQE-DR∥²
    )
    save_training_nqedr_output(output)

    # 打印总结
    logger.info("\n===== 训练总结 =====")
    logger.info(f"平均验证准确率: {np.mean(best_validation_accuracies):.4f} ± {np.std(best_validation_accuracies):.4f}")
    logger.info(f"最佳正则化参数: λ1={best_lamb1}, λ2={best_lamb2}")
    logger.info(f"最终Lipschitz界范围: [{np.min(lipschitz_bound_history):.4f}, {np.max(lipschitz_bound_history):.4f}]")

    return output


if __name__ == "__main__":
    run(dataset_name="circle")
    
