from pennylane import numpy as np
import matplotlib.pyplot as plt
# 加载四种编码方式各自的训练输出
from .file import (
    load_nqedr_training_output,    # NQE-DR编码
    load_hee_training_output,      # HEE编码
    load_ae_training_output,       # AE编码
    load_ame_training_output       # AmE编码
)

def plot_encoding_comparison():
    """绘制四种编码方式(AE, NQE-DR, AmE, HEE)的损失函数和准确率对比图"""
    # 加载每种编码方式的训练输出
    # NQE-DR编码
    nqedr_output = load_nqedr_training_output()
    nqedr_costs = nqedr_output.cost_over_epochs
    nqedr_train_accs = nqedr_output.train_accuracy_over_epochs
    nqedr_val_accs = nqedr_output.validation_accuracy_over_epochs
    
    # HEE编码
    hee_output = load_hee_training_output()
    hee_costs = hee_output.cost_over_epochs
    hee_train_accs = hee_output.train_accuracy_over_epochs
    hee_val_accs = hee_output.validation_accuracy_over_epochs
    
    # AE编码
    ae_output = load_ae_training_output()
    ae_costs = ae_output.cost_over_epochs
    ae_train_accs = ae_output.train_accuracy_over_epochs
    ae_val_accs = ae_output.validation_accuracy_over_epochs
    
    # AmE编码
    ame_output = load_ame_training_output()
    ame_costs = ame_output.cost_over_epochs
    ame_train_accs = ame_output.train_accuracy_over_epochs
    ame_val_accs = ame_output.validation_accuracy_over_epochs

    # NQE-DR编码
    nqedr_costs_first_lambda = nqedr_costs[0]
    nqedr_train_accs_first_lambda = nqedr_train_accs[0]
    nqedr_val_accs_first_lambda = nqedr_val_accs[0]
    
    # HEE编码
    hee_costs_first_lambda = hee_costs[0]
    hee_train_accs_first_lambda = hee_train_accs[0]
    hee_val_accs_first_lambda = hee_val_accs[0]
    
    # AE编码
    ae_costs_first_lambda = ae_costs[0]
    ae_train_accs_first_lambda = ae_train_accs[0]
    ae_val_accs_first_lambda = ae_val_accs[0]
    
    # AmE编码
    ame_costs_first_lambda = ame_costs[0]
    ame_train_accs_first_lambda = ame_train_accs[0]
    ame_val_accs_first_lambda = ame_val_accs[0]

    # 计算各编码方式的平均损失值和准确率（多轮运行的平均值）
    # NQE-DR编码
    nqedr_avg_costs = np.mean(nqedr_costs_first_lambda, axis=0)
    nqedr_avg_train_accs = np.mean(nqedr_train_accs_first_lambda, axis=0)
    nqedr_avg_val_accs = np.mean(nqedr_val_accs_first_lambda, axis=0)
    
    # HEE编码
    hee_avg_costs = np.mean(hee_costs_first_lambda, axis=0)
    hee_avg_train_accs = np.mean(hee_train_accs_first_lambda, axis=0)
    hee_avg_val_accs = np.mean(hee_val_accs_first_lambda, axis=0)
    
    # AE编码
    ae_avg_costs = np.mean(ae_costs_first_lambda, axis=0)
    ae_avg_train_accs = np.mean(ae_train_accs_first_lambda, axis=0)
    ae_avg_val_accs = np.mean(ae_val_accs_first_lambda, axis=0)
    
    # AmE编码
    ame_avg_costs = np.mean(ame_costs_first_lambda, axis=0)
    ame_avg_train_accs = np.mean(ame_train_accs_first_lambda, axis=0)
    ame_avg_val_accs = np.mean(ame_val_accs_first_lambda, axis=0)

    # 确定绘图的 epoch 范围（取所有编码中最大的epoch数）
    all_cost_arrays = [
        nqedr_avg_costs, 
        hee_avg_costs, 
        ae_avg_costs, 
        ame_avg_costs
    ]
    max_epochs = max([len(arr) for arr in all_cost_arrays])
    epochs = range(1, max_epochs + 1)

    # 绘制并保存损失函数对比曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs[:len(nqedr_avg_costs)], nqedr_avg_costs, 
             label='NQE-DR', color='tab:blue', linestyle='--', linewidth=2)
    plt.plot(epochs[:len(hee_avg_costs)], hee_avg_costs, 
             label='HEE', color='tab:green', linewidth=2)
    plt.plot(epochs[:len(ae_avg_costs)], ae_avg_costs, 
             label='AE', color='tab:red', linestyle='-.', linewidth=2)
    plt.plot(epochs[:len(ame_avg_costs)], ame_avg_costs, 
             label='AmE', color='tab:orange', linestyle=':', linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Comparison Across Encoding Methods", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./plots/encoding_loss_comparison.pdf", dpi=300)
    plt.close()

    # 绘制并保存训练准确率对比曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs[:len(nqedr_avg_train_accs)], nqedr_avg_train_accs, 
             label='NQE-DR (Train)', color='tab:blue', linestyle='--', linewidth=2)
    plt.plot(epochs[:len(hee_avg_train_accs)], hee_avg_train_accs, 
             label='HEE (Train)', color='tab:green', linewidth=2)
    plt.plot(epochs[:len(ae_avg_train_accs)], ae_avg_train_accs, 
             label='AE (Train)', color='tab:red', linestyle='-.', linewidth=2)
    plt.plot(epochs[:len(ame_avg_train_accs)], ame_avg_train_accs, 
             label='AmE (Train)', color='tab:orange', linestyle=':', linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training Accuracy Comparison Across Encoding Methods", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./plots/encoding_training_accuracy.pdf", dpi=300)
    plt.close()

    # 绘制并保存验证准确率对比曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs[:len(nqedr_avg_val_accs)], nqedr_avg_val_accs, 
             label='NQE-DR (Validation)', color='tab:blue', linestyle='--', linewidth=2)
    plt.plot(epochs[:len(hee_avg_val_accs)], hee_avg_val_accs, 
             label='HEE (Validation)', color='tab:green', linewidth=2)
    plt.plot(epochs[:len(ae_avg_val_accs)], ae_avg_val_accs, 
             label='AE (Validation)', color='tab:red', linestyle='-.', linewidth=2)
    plt.plot(epochs[:len(ame_avg_val_accs)], ame_avg_val_accs, 
             label='AmE (Validation)', color='tab:orange', linestyle=':', linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation Accuracy Comparison Across Encoding Methods", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./plots/encoding_validation_accuracy.pdf", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_encoding_comparison()
