import gradio as gr
import torch
import numpy as np
import cv2
from unet import UNet
from config import PredictConfig


# 加载模型
def load_trained_model():
    model = UNet(n_channels=PredictConfig.N_CHANNELS, n_classes=PredictConfig.N_CLASSES)
    model.load_state_dict(torch.load(PredictConfig.MODEL_PATH, map_location=PredictConfig.DEVICE))
    model.to(PredictConfig.DEVICE)
    model.eval()
    return model


# 上传后对图像进行预处理（用于显示 + 后续预测）
def preprocess_uploaded_image(image):
    if image.ndim == 2:
        # 如果是单通道灰度图，将其转为 3 通道 RGB 用于展示
        display_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        processed = display_image.copy()  # 用于后续模型推理
    elif image.ndim == 3:
        # 如果是多通道图像，截取前 3 个通道（RGB）
        if image.shape[2] > 3:
            processed = image[:, :, :3]
        else:
            processed = image
        display_image = processed.astype(np.uint8)
    else:
        raise ValueError("图像格式不支持")
    return display_image, processed.astype(np.uint8)  # 返回展示图像和模型输入图像


# 预测函数，使用预处理后的图像进行 UNet 分割
def predict(processed_image):
    model = load_trained_model()

    # 将 RGB 图像转换为灰度图
    gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
    tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float().to(PredictConfig.DEVICE)

    # 推理阶段不计算梯度
    with torch.no_grad():
        output = model(tensor)[0][0].cpu().numpy()

    output = (output >= PredictConfig.THRESHOLD).astype(np.uint8) * 255
    return output


# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 👁️ UNet 血管图像分割 - 预测工具")

    state = gr.State()  # 全局变量，用于存储上传后处理好的图像，供预测使用

    with gr.Row():
        img_input = gr.Image(type="numpy", label="上传图像", image_mode="RGB", interactive=True)
        img_output = gr.Image(type="numpy", label="分割结果")

    upload_info = gr.Markdown("")  # 提示上传状态


    # 上传后立即处理图像
    def handle_upload(image):
        display_img, processed_img = preprocess_uploaded_image(image)
        return display_img, processed_img, "✅ 图像上传成功，点击下方按钮进行预测"


    img_input.upload(fn=handle_upload, inputs=img_input, outputs=[img_input, state, upload_info])

    predict_btn = gr.Button("开始预测")
    predict_btn.click(fn=predict, inputs=state, outputs=img_output)

demo.launch()
