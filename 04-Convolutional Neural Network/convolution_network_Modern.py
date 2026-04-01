import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import time

# 1. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 2. 超参数与数据加载
batch_size = 128
learning_rate = 1e-2
num_epoches = 10

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. 定义模型 (Flatten 已整合)
class ModernCnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(ModernCnn, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),  # 自动把 [B, 16, 5, 5] 压成 [B, 400]
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        return self.model(x)


# 4. 初始化
model = ModernCnn(1, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --- 关键：创建 TensorBoard 写入器 ---
writer = SummaryWriter("logs_mnist")  # 会在项目下创建 logs_mnist 文件夹

if __name__ == '__main__':
    print(f"当前使用的设备是: {device}")
    total_train_step = 0

    for epoch in range(num_epoches):
        model.train()
        print(f"----- 第 {epoch + 1} 轮训练开始 -----")

        for data in train_loader:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                # (1) 记录标量: Loss
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                print(f"训练次数: {total_train_step}, 损失: {loss.item():.6f}")

        # --- 测试与准确率记录 ---
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                total_correct += (outputs.argmax(1) == labels).sum().item()

        acc = total_correct / len(test_dataset)
        # (2) 记录标量: Accuracy
        writer.add_scalar("test_accuracy", acc, epoch)

        # (3) 记录图片: 看看模型在看什么
        # 取出一个 batch 的图片预览
        img_grid = torchvision.utils.make_grid(imgs[:16])
        writer.add_image("mnist_images_preview", img_grid, epoch)

    writer.close()  # 记得关闭
    print("训练结束，请在终端输入: tensorboard --logdir=logs_mnist 查看结果")
