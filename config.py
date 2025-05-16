import torch


class TrainConfig:
    # ================== 数据路径 ==================
    DATA_PATH: str = "data/train/"  # 训练数据目录

    # ================== 模型训练参数 ==================
    EPOCHS: int = 10  # 训练轮数
    BATCH_SIZE: int = 1  # 每批次训练样本数
    LEARNING_RATE: float = 1e-5  # 学习率

    # ================== 优化器参数 ==================
    WEIGHT_DECAY: float = 1e-8  # 权重衰减
    MOMENTUM: float = 0.9  # 动量

    # ================== 模型保存设置 ==================
    MODEL_SAVE_PATH: str = "best_model.pth"  # 最佳模型保存路径

    # ================== 运行设备设置 ==================
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================== 日志输出控制 ==================
    VERBOSE: bool = False  # 是否输出详细日志
    PRINT_INTERVAL: int = 5  # 每 N 个 batch 输出一次 loss


class PredictConfig:
    # ================== 模型与路径设置 ==================
    MODEL_PATH: str = "best_model.pth"  # 加载的模型路径
    TEST_PATH: str = "data/test/images/*.tif"  # 测试图像路径通配符
    SAVE_PATH: str = "data/test/results"  # 预测结果保存路径

    # ================== 图像参数 ==================
    N_CHANNELS: int = 1  # 输入通道数
    N_CLASSES: int = 1  # 输出通道数

    # ================== 预测控制参数 ==================
    THRESHOLD: float = 0.5  # 二值化阈值
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================== 图像可视化参数 ==================
    FIGURE_SIZE: tuple = (10, 5)
    TITLES: dict = {
        'original': 'Original Image',
        'predicted': 'Predicted Mask'
    }
