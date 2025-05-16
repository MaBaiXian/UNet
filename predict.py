import glob
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from unet import UNet
from config import PredictConfig


def load_model():
    # 初始化模型
    model = UNet(n_channels=PredictConfig.N_CHANNELS, n_classes=PredictConfig.N_CLASSES)
    model.to(PredictConfig.DEVICE)
    try:
        # 加载训练好的模型参数
        model.load_state_dict(torch.load(PredictConfig.MODEL_PATH, map_location=PredictConfig.DEVICE))
        print(f"✔ Loaded model from {PredictConfig.MODEL_PATH}")
    except Exception as e:
        # 加载失败时抛出异常
        raise RuntimeError(f"❌ Failed to load model: {e}")
    model.eval()
    return model


def preprocess_image(path):
    # 读取图像，灰度模式
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"⚠️ Cannot read image: {path}")
    img = img.reshape(1, 1, img.shape[0], img.shape[1])  # [B, C, H, W]
    # 转为tensor并移动到设备上
    img_tensor = torch.from_numpy(img).float().to(PredictConfig.DEVICE)
    return img_tensor, img[0][0]  # 返回原图用于可视化


def postprocess_prediction(pred_tensor):
    # 从预测tensor转换为numpy数组
    pred = pred_tensor.data.cpu().numpy()[0][0]
    # 二值化处理：大于等于阈值为255，小于为0
    pred = (pred >= PredictConfig.THRESHOLD).astype(np.uint8) * 255
    return pred


def save_and_show_result(original, pred, save_path, file_name):
    plt.figure(figsize=PredictConfig.FIGURE_SIZE)
    plt.subplot(1, 2, 1)
    plt.title(PredictConfig.TITLES['original'])
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(PredictConfig.TITLES['predicted'])
    plt.imshow(pred, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, file_name)
    # 保存图像
    plt.savefig(full_path, bbox_inches='tight')
    print(f"✅ Saved result to {full_path}")
    plt.show()


def main():
    model = load_model()  # 加载模型
    test_image_paths = glob.glob(PredictConfig.TEST_PATH)  # 获取所有测试图像路径

    # 遍历每张测试图像进行预测
    if not test_image_paths:
        print("❌ No test images found.")
        return

    for path in test_image_paths:
        try:
            # 图像预处理
            img_tensor, original = preprocess_image(path)

            # 模型预测
            pred_tensor = model(img_tensor)

            # 后处理预测结果
            pred_mask = postprocess_prediction(pred_tensor)

            # 构建保存文件名并保存图像
            file_name = os.path.splitext(os.path.basename(path))[0] + "_result.png"
            save_and_show_result(original, pred_mask, PredictConfig.SAVE_PATH, file_name)
        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")


if __name__ == "__main__":
    main()
