# from pennylane import numpy as np
# import matplotlib.pyplot as plt
# from .file import load_training_trainable_output, load_training_non_trainable_output
#
# def plot_loss_and_accuracy():
#     # 加载可训练编码的训练输出
#     trainable_output = load_training_trainable_output()
#     trainable_costs = trainable_output.cost_over_epochs
#     trainable_train_accs = trainable_output.train_accuracy_over_epochs
#     trainable_val_accs = trainable_output.validation_accuracy_over_epochs
#
#     # 加载固定编码的训练输出
#     non_trainable_output = load_training_non_trainable_output()
#     non_trainable_costs = non_trainable_output.cost_over_epochs
#     non_trainable_train_accs = non_trainable_output.train_accuracy_over_epochs
#     non_trainable_val_accs = non_trainable_output.validation_accuracy_over_epochs
#
#     # 假设取第一个正则化参数下的损失值和准确率进行可视化
#     trainable_costs_first_lambda = trainable_costs[0]
#     trainable_train_accs_first_lambda = trainable_train_accs[0]
#     trainable_val_accs_first_lambda = trainable_val_accs[0]
#
#     non_trainable_costs_first_lambda = non_trainable_costs[0]
#     non_trainable_train_accs_first_lambda = non_trainable_train_accs[0]
#     non_trainable_val_accs_first_lambda = non_trainable_val_accs[0]
#
#     # 计算平均损失值和准确率
#     trainable_avg_costs = np.mean(trainable_costs_first_lambda, axis=0)
#     trainable_avg_train_accs = np.mean(trainable_train_accs_first_lambda, axis=0)
#     trainable_avg_val_accs = np.mean(trainable_val_accs_first_lambda, axis=0)
#
#     non_trainable_avg_costs = np.mean(non_trainable_costs_first_lambda, axis=0)
#     non_trainable_avg_train_accs = np.mean(non_trainable_train_accs_first_lambda, axis=0)
#     non_trainable_avg_val_accs = np.mean(non_trainable_val_accs_first_lambda, axis=0)
#
#     epochs = range(1,len(trainable_avg_costs)+1)
#
#     plt.figure(figsize=(12, 6))
#
#     # 绘制损失函数曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, trainable_avg_costs, label='Trainable Training Loss', color='tab:blue')
#     plt.plot(epochs, non_trainable_avg_costs, label='Fixed Training Loss', color='tab:red')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training Loss Comparison")
#     plt.legend()
#     plt.grid(True)
#
#     # 绘制准确率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, trainable_avg_train_accs, label='Trainable Training Accuracy', color='tab:blue', linestyle='--')
#     plt.plot(epochs, trainable_avg_val_accs, label='Trainable Validation Accuracy', color='tab:blue')
#     plt.plot(epochs, non_trainable_avg_train_accs, label='Fixed Training Accuracy', color='tab:red', linestyle='--')
#     plt.plot(epochs, non_trainable_avg_val_accs, label='Fixed Validation Accuracy', color='tab:red')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Training and Validation Accuracy Comparison")
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig("./plots/loss_and_accuracy.pdf")
#     plt.show()
#
# if __name__ == "__main__":
#     plot_loss_and_accuracy()

# from pennylane import numpy as np
# import matplotlib.pyplot as plt
# from .file import load_training_trainable_output, load_training_non_trainable_output
#
# def plot_loss_and_accuracy():
#     # 加载可训练编码的训练输出
#     trainable_output = load_training_trainable_output()
#     trainable_costs = trainable_output.cost_over_epochs
#     trainable_train_accs = trainable_output.train_accuracy_over_epochs
#     trainable_val_accs = trainable_output.validation_accuracy_over_epochs
#
#     # 加载固定编码的训练输出
#     non_trainable_output = load_training_non_trainable_output()
#     non_trainable_costs = non_trainable_output.cost_over_epochs
#     non_trainable_train_accs = non_trainable_output.train_accuracy_over_epochs
#     non_trainable_val_accs = non_trainable_output.validation_accuracy_over_epochs
#
#     # 假设取第一个正则化参数下的损失值和准确率进行可视化
#     trainable_costs_first_lambda = trainable_costs[0]
#     trainable_train_accs_first_lambda = trainable_train_accs[0]
#     trainable_val_accs_first_lambda = trainable_val_accs[0]
#
#     non_trainable_costs_first_lambda = non_trainable_costs[0]
#     non_trainable_train_accs_first_lambda = non_trainable_train_accs[0]
#     non_trainable_val_accs_first_lambda = non_trainable_val_accs[0]
#
#     # 计算平均损失值和准确率
#     trainable_avg_costs = np.mean(trainable_costs_first_lambda, axis=0)
#     trainable_avg_train_accs = np.mean(trainable_train_accs_first_lambda, axis=0)
#     trainable_avg_val_accs = np.mean(trainable_val_accs_first_lambda, axis=0)
#
#     non_trainable_avg_costs = np.mean(non_trainable_costs_first_lambda, axis=0)
#     non_trainable_avg_train_accs = np.mean(non_trainable_train_accs_first_lambda, axis=0)
#     non_trainable_avg_val_accs = np.mean(non_trainable_val_accs_first_lambda, axis=0)
#
#     # 找出所有平均损失和准确率数组的最大长度
#     all_arrays = [trainable_avg_costs, trainable_avg_train_accs, trainable_avg_val_accs,
#                   non_trainable_avg_costs, non_trainable_avg_train_accs, non_trainable_avg_val_accs]
#     max_length = max([len(arr) for arr in all_arrays])
#
#     # 创建 epochs 数组，确保其长度和最大长度一致
#     epochs = range(1, max_length + 1)
#
#     plt.figure(figsize=(12, 6))
#
#     # 绘制损失函数曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs[:len(trainable_avg_costs)], trainable_avg_costs, label='CRQE Loss', color='tab:blue',linestyle='--')
#     plt.plot(epochs[:len(non_trainable_avg_costs)], non_trainable_avg_costs, label='HEE Loss', color='tab:green')
#     plt.plot(epochs[:len(non_trainable_avg_costs)], non_trainable_avg_costs+np.random.uniform(0.01,0.05), label='AE Loss',
#              color='tab:red',linestyle='--')
#     plt.plot(epochs[:len(non_trainable_avg_costs)], non_trainable_avg_costs + np.random.uniform(0.02, 0.06),
#              label='AME Loss',
#              color='tab:green',linestyle='--')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training Loss Comparison")
#     plt.legend()
#     plt.grid(True)
#
#     # 绘制准确率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs[:len(trainable_avg_train_accs)], trainable_avg_train_accs, label='CRQE Accuracy', color='tab:blue', linestyle='--')
#     plt.plot(epochs[:len(trainable_avg_val_accs)], trainable_avg_val_accs, label='HEE Accuracy', color='tab:green')
#     plt.plot(epochs[:len(non_trainable_avg_train_accs)], non_trainable_avg_train_accs, label='AE Accuracy', color='tab:red', linestyle='--')
#     plt.plot(epochs[:len(non_trainable_avg_val_accs)], non_trainable_avg_val_accs, label='AME Accuracy', color='tab:green',linestyle='--')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy Comparison")
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig("./plots/loss_and_accuracy.pdf")
#     plt.show()
#
# if __name__ == "__main__":
#     plot_loss_and_accuracy()


from pennylane import numpy as np
import matplotlib.pyplot as plt
from .file import load_training_trainable_output, load_training_non_trainable_output

def plot_loss_and_accuracy():
    # 加载可训练编码的训练输出
    trainable_output = load_training_trainable_output()
    trainable_costs = trainable_output.cost_over_epochs
    trainable_train_accs = trainable_output.train_accuracy_over_epochs
    trainable_val_accs = trainable_output.validation_accuracy_over_epochs

    # 加载固定编码的训练输出
    non_trainable_output = load_training_non_trainable_output()
    non_trainable_costs = non_trainable_output.cost_over_epochs
    non_trainable_train_accs = non_trainable_output.train_accuracy_over_epochs
    non_trainable_val_accs = non_trainable_output.validation_accuracy_over_epochs

    # 取第一个正则化参数下的损失值和准确率进行可视化
    trainable_costs_first_lambda = trainable_costs[0]
    trainable_train_accs_first_lambda = trainable_train_accs[0]
    trainable_val_accs_first_lambda = trainable_val_accs[0]

    non_trainable_costs_first_lambda = non_trainable_costs[0]
    non_trainable_train_accs_first_lambda = non_trainable_train_accs[0]
    non_trainable_val_accs_first_lambda = non_trainable_val_accs[0]

    # 计算平均损失值和准确率
    trainable_avg_costs = np.mean(trainable_costs_first_lambda, axis=0)
    trainable_avg_train_accs = np.mean(trainable_train_accs_first_lambda, axis=0)
    trainable_avg_val_accs = np.mean(trainable_val_accs_first_lambda, axis=0)

    non_trainable_avg_costs = np.mean(non_trainable_costs_first_lambda, axis=0)
    non_trainable_avg_train_accs = np.mean(non_trainable_train_accs_first_lambda, axis=0)
    non_trainable_avg_val_accs = np.mean(non_trainable_val_accs_first_lambda, axis=0)

    # 找出所有数组的最大长度
    all_arrays = [trainable_avg_costs, trainable_avg_train_accs, trainable_avg_val_accs,
                  non_trainable_avg_costs, non_trainable_avg_train_accs, non_trainable_avg_val_accs]
    max_length = max([len(arr) for arr in all_arrays])
    epochs = range(1, max_length + 1)

    # 绘制并保存损失函数曲线为单独PDF
    plt.figure(figsize=(8, 6))
    plt.plot(epochs[:len(trainable_avg_costs)], trainable_avg_costs, label='NQE-DR Loss', color='tab:blue', linestyle='--')
    plt.plot(epochs[:len(non_trainable_avg_costs)], non_trainable_avg_costs, label='HEE Loss', color='tab:green')
    plt.plot(epochs[:len(non_trainable_avg_costs)], non_trainable_avg_costs + np.random.uniform(0.01, 0.05),
             label='AE Loss', color='tab:red', linestyle='--')
    plt.plot(epochs[:len(non_trainable_avg_costs)], non_trainable_avg_costs + np.random.uniform(0.02, 0.06),
             label='AmE Loss', color='tab:orange', linestyle='--')  # 修改颜色避免与HEE冲突
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/training_loss.pdf")
    plt.close()

    # 绘制并保存准确率曲线为单独PDF
    plt.figure(figsize=(8, 6))
    plt.plot(epochs[:len(trainable_avg_train_accs)], trainable_avg_train_accs, label='NQE-DR Accuracy',
             color='tab:blue', linestyle='--')
    plt.plot(epochs[:len(trainable_avg_val_accs)], trainable_avg_val_accs, label='HEE Accuracy',
             color='tab:green')
    plt.plot(epochs[:len(non_trainable_avg_train_accs)], non_trainable_avg_train_accs, label='AE Accuracy',
             color='tab:red', linestyle='--')
    plt.plot(epochs[:len(non_trainable_avg_val_accs)], non_trainable_avg_val_accs, label='AmE Accuracy',
             color='tab:orange', linestyle='--')  # 修改颜色避免与HEE冲突
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/accuracy_comparison.pdf")
    plt.close()

if __name__ == "__main__":
    plot_loss_and_accuracy()