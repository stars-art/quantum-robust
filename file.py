from pickle import dump, load
import os

# 确保数据目录存在
def ensure_dir():
    if not os.path.exists("./data"):
        os.makedirs("./data")

# ==============================
# NQE-DR 编码方式的文件操作
# ==============================
def save_training_nqedr_output(output):
    """保存NQE-DR编码模型的训练输出"""
    ensure_dir()
    dump(output, open("./data/train_nqedr.p", "wb"))
    return

def load_training_nqedr_output():
    """加载NQE-DR编码模型的训练输出"""
    return load(open("./data/train_nqedr.p", "rb"))

def save_nqedr_robustness_output(output):
    """保存NQE-DR编码模型的鲁棒性输出"""
    ensure_dir()
    dump(output, open("./data/robustness_nqedr.p", "wb"))
    return

def load_nqedr_robustness_output():
    """加载NQE-DR编码模型的鲁棒性输出"""
    return load(open("./data/robustness_nqedr.p", "rb"))

def save_nqedr_generalization_output(output):
    """保存NQE-DR编码模型的泛化能力输出"""
    ensure_dir()
    dump(output, open("./data/generalization_nqedr.p", "wb"))
    return

def load_nqedr_generalization_output():
    """加载NQE-DR编码模型的泛化能力输出"""
    return load(open("./data/generalization_nqedr.p", "rb"))

# ==============================
# HEE 编码方式的文件操作
# ==============================
def save_training_hee_output(output):
    """保存HEE编码模型的训练输出"""
    ensure_dir()
    dump(output, open("./data/train_hee.p", "wb"))
    return

def load_training_hee_output():
    """加载HEE编码模型的训练输出"""
    return load(open("./data/train_hee.p", "rb"))

def save_hee_robustness_output(output):
    """保存HEE编码模型的鲁棒性输出"""
    ensure_dir()
    dump(output, open("./data/robustness_hee.p", "wb"))
    return

def load_hee_robustness_output():
    """加载HEE编码模型的鲁棒性输出"""
    return load(open("./data/robustness_hee.p", "rb"))

def save_hee_generalization_output(output):
    """保存HEE编码模型的泛化能力输出"""
    ensure_dir()
    dump(output, open("./data/generalization_hee.p", "wb"))
    return

def load_hee_generalization_output():
    """加载HEE编码模型的泛化能力输出"""
    return load(open("./data/generalization_hee.p", "rb"))

# ==============================
# AE 编码方式的文件操作
# ==============================
def save_training_ae_output(output):
    """保存AE编码模型的训练输出"""
    ensure_dir()
    dump(output, open("./data/train_ae.p", "wb"))
    return

def load_training_ae_output():
    """加载AE编码模型的训练输出"""
    return load(open("./data/train_ae.p", "rb"))

def save_ae_robustness_output(output):
    """保存AE编码模型的鲁棒性输出"""
    ensure_dir()
    dump(output, open("./data/robustness_ae.p", "wb"))
    return

def load_ae_robustness_output():
    """加载AE编码模型的鲁棒性输出"""
    return load(open("./data/robustness_ae.p", "rb"))

def save_ae_generalization_output(output):
    """保存AE编码模型的泛化能力输出"""
    ensure_dir()
    dump(output, open("./data/generalization_ae.p", "wb"))
    return

def load_ae_generalization_output():
    """加载AE编码模型的泛化能力输出"""
    return load(open("./data/generalization_ae.p", "rb"))

# ==============================
# AmE 编码方式的文件操作
# ==============================
def save_training_ame_output(output):
    """保存AmE编码模型的训练输出"""
    ensure_dir()
    dump(output, open("./data/train_ame.p", "wb"))
    return

def load_training_ame_output():
    """加载AmE编码模型的训练输出"""
    return load(open("./data/train_ame.p", "rb"))

def save_ame_robustness_output(output):
    """保存AmE编码模型的鲁棒性输出"""
    ensure_dir()
    dump(output, open("./data/robustness_ame.p", "wb"))
    return

def load_ame_robustness_output():
    """加载AmE编码模型的鲁棒性输出"""
    return load(open("./data/robustness_ame.p", "rb"))

def save_ame_generalization_output(output):
    """保存AmE编码模型的泛化能力输出"""
    ensure_dir()
    dump(output, open("./data/generalization_ame.p", "wb"))
    return

def load_ame_generalization_output():
    """加载AmE编码模型的泛化能力输出"""
    return load(open("./data/generalization_ame.p", "rb"))

