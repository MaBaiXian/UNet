import matplotlib.pyplot as plt
from unet import UNet
from dataset import DRIVE_Loader
from torch import optim
import torch.nn as nn
import torch
from config import TrainConfig


# 定义训练函数
def train_net(net, train_loader, optimizer, criterion):
    loss_values = []  # 存储每个 epoch 的平均损失
    best_loss = float('inf')  # 初始化最优损失为正无穷，用于模型保存判断

    for epoch in range(TrainConfig.EPOCHS):
        net.train()  # 设置模型为训练模式
        epoch_loss = 0.0  # 当前 epoch 的损失累计

        # 遍历所有训练 batch
        for batch_idx, (image, label) in enumerate(train_loader):
            # 将图像和标签转移到训练设备上（如 GPU）
            image = image.to(device=TrainConfig.DEVICE, dtype=torch.float32)
            label = label.to(device=TrainConfig.DEVICE, dtype=torch.float32)

            optimizer.zero_grad()  # 清除前一次梯度
            pred = net(image)  # 前向传播，得到预测结果
            loss = criterion(pred, label)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            epoch_loss += loss.item()  # 累计 batch 损失

            # 可选打印每 N 个 batch 的损失值（用于调试）
            if TrainConfig.VERBOSE and (batch_idx + 1) % TrainConfig.PRINT_INTERVAL == 0:
                print(f"[Epoch {epoch + 1}] Batch {batch_idx + 1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")

        # 计算当前 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_epoch_loss)  # 记录当前 epoch 损失

        # 如果当前损失优于历史最优损失，则保存模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(net.state_dict(), TrainConfig.MODEL_SAVE_PATH)
            print(f"✔ Saved best model at epoch {epoch + 1} with loss {best_loss:.4f}")

        # 输出当前 epoch 的平均损失
        print(f"Epoch [{epoch + 1}/{TrainConfig.EPOCHS}] - Loss: {avg_epoch_loss:.4f}")

    return loss_values  # 返回所有 epoch 的损失列表


# 绘制训练损失变化图
def plot_loss(loss_values):
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', color='b', label='Training Loss')
    plt.title('Training Loss over Epochs')  # 图标题
    plt.xlabel('Epochs')  # 横轴标签
    plt.ylabel('Loss')  # 纵轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 网格
    plt.savefig('training_loss.png')  # 保存图像为 PNG 文件
    plt.show()  # 显示图像


# 主程序入口
def main():
    try:
        # 初始化数据集和数据加载器
        dataset = DRIVE_Loader(TrainConfig.DATA_PATH)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=True
        )

        # 构建 UNet 模型并移至指定设备
        net = UNet(n_channels=1, n_classes=1).to(TrainConfig.DEVICE)

        # 设置优化器（RMSprop）
        optimizer = optim.RMSprop(
            net.parameters(),
            lr=TrainConfig.LEARNING_RATE,
            weight_decay=TrainConfig.WEIGHT_DECAY,
            momentum=TrainConfig.MOMENTUM
        )

        # 设置损失函数（二分类，使用 BCE with Logits）
        criterion = nn.BCEWithLogitsLoss()

        # 开始训练并获取每轮的损失值
        loss_values = train_net(net, train_loader, optimizer, criterion)

        # 绘制损失曲线
        plot_loss(loss_values)

    # 键盘中断时输出提示
    except KeyboardInterrupt:
        print("❗ Training interrupted.")

    # 其他异常情况输出错误信息
    except Exception as e:
        print(f"⚠️ Training failed: {e}")

    # 最后释放显存，防止内存泄露
    finally:
        torch.cuda.empty_cache()


# 程序主入口
if __name__ == "__main__":
    main()
