from dataclasses import dataclass
import numpy as np

# @dataclass
# class TrainingOutput:
#     x_train: np.ndarray  # 训练数据点，形状为(n_training, features)
#     y_train: np.ndarray  # 训练标签，形状为(n_training,)
#     x_validation: np.ndarray  # 验证数据点，形状为(n_validation, features)
#     y_validation: np.ndarray  # 验证标签，形状为(n_validation,)
#     lambdas: np.ndarray  # 正则化参数数组，形状为(n_lambda,)
#     cost_over_epochs: np.ndarray  # 损失随轮次变化，形状为(n_lambda, n_ensemble, n_epoch)
#     train_accuracy_over_epochs: np.ndarray  # 训练准确率，形状同上
#     validation_accuracy_over_epochs: np.ndarray  # 验证准确率，形状同上
#     weights_over_epochs: np.ndarray  # 权重参数，形状为(n_lambda, n_ensemble, n_epoch, n_layer, n_qubit, x_dim, x_dim)
#     biases_over_epochs: np.ndarray  # 偏置参数，形状为(n_lambda, n_ensemble, n_epoch, n_layer, n_qubit, x_dim)
#     lipschitz_bound_over_epochs: np.ndarray  # Lipschitz界，形状为(n_lambda, n_ensemble, n_epoch)



# @dataclass
# class TrainingNonTrainableOutput:
#     """非训练模型训练输出（保留兼容性）"""
#     x_train: np.ndarray
#     y_train: np.ndarray
#     x_validation: np.ndarray
#     y_validation: np.ndarray
#     lambdas: np.ndarray
#     cost_over_epochs: np.ndarray
#     train_accuracy_over_epochs: np.ndarray
#     validation_accuracy_over_epochs: np.ndarray
#     weights_over_epochs: np.ndarray

# @dataclass
# class RobustnessOutput:
#     """基础模型鲁棒性输出（保留兼容性）"""
#     lambdas: np.ndarray  # 正则化参数数组
#     epsilons: np.ndarray  # 噪声水平数组
#     test_accuracies: np.ndarray  # 测试准确率，形状为(n_lambda, n_noise, n_ensemble)
#     lipschitz_bounds: np.ndarray  # Lipschitz界
#     x_test: np.ndarray  # 测试数据点
#     y_test: np.ndarray  # 测试标签

# @dataclass
# class NonTrainableRobustnessOutput:
#     """非训练模型鲁棒性输出（保留兼容性）"""
#     lambdas: np.ndarray
#     epsilons: np.ndarray
#     test_accuracies: np.ndarray
#     x_test: np.ndarray
#     y_test: np.ndarray



@dataclass
class NQEDRTrainingOutput:
    """NQE-DR编码模型训练输出"""
    x_train: np.ndarray  # 训练数据点，形状为(n_training, features)
    y_train: np.ndarray  # 训练标签，形状为(n_training,)
    x_validation: np.ndarray  # 验证数据点，形状为(n_validation, features)
    y_validation: np.ndarray  # 验证标签，形状为(n_validation,)
    best_lambda1: float  # 最佳参数正则化强度
    best_lambda2: float  # 最佳梯度正则化强度
    cost_over_epochs: np.ndarray  # 损失随轮次变化，形状为(n_ensemble, n_epoch)
    train_accuracy_over_epochs: np.ndarray  # 训练准确率，形状同上
    validation_accuracy_over_epochs: np.ndarray  # 验证准确率，形状同上
    weights_over_epochs: np.ndarray  # 权重参数，形状为(n_ensemble, n_layer, n_qubit, x_dim, x_dim)
    biases_over_epochs: np.ndarray  # 偏置参数，形状为(n_ensemble, n_layer, n_qubit, x_dim)
    param_regularization_over_epochs: np.ndarray  # 参数正则化项，形状为(n_ensemble, n_epoch)
    lipschitz_regularization_over_epochs: np.ndarray  # Lipschitz正则化项，形状同上

@dataclass
class NQEDRRobustnessOutput:
    """NQE-DR编码模型鲁棒性输出"""
    best_lambda1: float  # 最佳参数正则化强度
    best_lambda2: float  # 最佳梯度正则化强度
    epsilons: np.ndarray  # 噪声水平数组，形状为(n_epsilon,)
    fgsm_epsilons: np.ndarray  # FGSM攻击强度数组
    noise_test_accuracies: np.ndarray  # 噪声下测试准确率，形状为(n_epsilon, n_ensemble)
    fgsm_test_accuracies: np.ndarray  # FGSM攻击下测试准确率，形状同上
    lipschitz_bounds: np.ndarray  # Lipschitz界，形状为(n_ensemble,)
    x_test: np.ndarray  # 测试数据点，形状为(n_test, features)
    y_test: np.ndarray  # 测试标签，形状为(n_test,)
    noise_levels: np.ndarray  # 噪声水平详细记录


@dataclass
class HEETrainingOutput:
    """HEE编码模型训练输出"""
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    best_lambda1: float  # 最佳参数正则化强度
    best_lambda2: float  # 最佳梯度正则化强度
    cost_over_epochs: np.ndarray  # 形状为(n_ensemble, n_epoch)
    train_accuracy_over_epochs: np.ndarray
    validation_accuracy_over_epochs: np.ndarray
    weights_over_epochs: np.ndarray  # 形状为(n_ensemble, n_layer, n_qubit, x_dim, x_dim)
    biases_over_epochs: np.ndarray  # 形状为(n_ensemble, n_layer, n_qubit, x_dim)
    param_regularization_over_epochs: np.ndarray
    lipschitz_regularization_over_epochs: np.ndarray

@dataclass
class HEERobustnessOutput:
    """HEE编码模型鲁棒性输出"""
    best_lambda1: float
    best_lambda2: float
    epsilons: np.ndarray
    fgsm_epsilons: np.ndarray
    noise_test_accuracies: np.ndarray
    fgsm_test_accuracies: np.ndarray
    lipschitz_bounds: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    noise_levels: np.ndarray


@dataclass
class AETrainingOutput:
    """AE编码模型训练输出"""
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    best_lambda1: float  # 最佳参数正则化强度
    best_lambda2: float  # 最佳梯度正则化强度
    cost_over_epochs: np.ndarray  # 形状为(n_ensemble, n_epoch)
    train_accuracy_over_epochs: np.ndarray
    validation_accuracy_over_epochs: np.ndarray
    weights_over_epochs: np.ndarray  # 形状为(n_ensemble, n_layer, n_qubit, x_dim, x_dim)
    biases_over_epochs: np.ndarray  # 形状为(n_ensemble, n_layer, n_qubit, x_dim)
    param_regularization_over_epochs: np.ndarray
    lipschitz_regularization_over_epochs: np.ndarray

@dataclass
class AERobustnessOutput:
    """AE编码模型鲁棒性输出"""
    best_lambda1: float
    best_lambda2: float
    epsilons: np.ndarray
    fgsm_epsilons: np.ndarray
    noise_test_accuracies: np.ndarray
    fgsm_test_accuracies: np.ndarray
    lipschitz_bounds: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    noise_levels: np.ndarray


@dataclass
class AMETrainingOutput:
    """AmE编码模型训练输出"""
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    best_lambda1: float  # 最佳参数正则化强度
    best_lambda2: float  # 最佳梯度正则化强度
    cost_over_epochs: np.ndarray  # 形状为(n_ensemble, n_epoch)
    train_accuracy_over_epochs: np.ndarray
    validation_accuracy_over_epochs: np.ndarray
    weights_over_epochs: np.ndarray  # 形状为(n_ensemble, n_layer, n_qubit, x_dim, x_dim)
    biases_over_epochs: np.ndarray  # 形状为(n_ensemble, n_layer, n_qubit, x_dim)
    param_regularization_over_epochs: np.ndarray
    lipschitz_regularization_over_epochs: np.ndarray

@dataclass
class AMERobustnessOutput:
    """AmE编码模型鲁棒性输出"""
    best_lambda1: float
    best_lambda2: float
    epsilons: np.ndarray
    fgsm_epsilons: np.ndarray
    noise_test_accuracies: np.ndarray
    fgsm_test_accuracies: np.ndarray
    lipschitz_bounds: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    noise_levels: np.ndarray


@dataclass
class EncodingComparisonOutput:
    """四种编码方式对比结果"""
    dataset_name: str  # 数据集名称
    encoding_methods: list  # 编码方式列表: ["NQE-DR", "HEE", "AE", "AmE"]
    clean_accuracies: np.ndarray  # 干净数据上的准确率，形状为(4,)
    noise_accuracies: np.ndarray  # 噪声下的准确率，形状为(4, n_epsilon)
    fgsm_accuracies: np.ndarray  # FGSM攻击下的准确率，形状为(4, n_epsilon)
    lipschitz_bounds: np.ndarray  # Lipschitz界，形状为(4,)
    training_times: np.ndarray  # 训练时间，形状为(4,)
    best_lambda1: dict  # 每种编码的最佳λ1: {编码方式: 值}
    best_lambda2: dict  # 每种编码的最佳λ2: {编码方式: 值}
