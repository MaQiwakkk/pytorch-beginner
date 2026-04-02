import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import os
import time

# 1. GPU设备配置
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 2. 超参数与目录准备
num_epochs = 50
batch_size = 128
learning_rate = 1e-3
log_dir = './logs_ae'
img_dir = './dc_img'

if not os.path.exists(img_dir):
    os.mkdir(img_dir)

# 3. 数据加载与预处理
# 注意：原版使用了 Normalize((0.5), (0.5))，这会将 [0,1] 映射到 [-1,1]
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 4. 定义卷积自编码器模型
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # Encoder: 压缩空间，提取特征
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # -> [B, 16, 10, 10]
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # -> [B, 16, 5, 5]
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # -> [B, 8, 3, 3]
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # -> [B, 8, 2, 2] (Latent Space)
        )
        # Decoder: 还原空间，重建像素
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # -> [B, 16, 5, 5]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # -> [B, 8, 15, 15]
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # -> [B, 1, 28, 28]
            nn.Tanh()  # 对应 Normalize 的 [-1, 1] 区间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 5. 初始化模型、损失函数和优化器
model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()  # AE 的核心就是计算像素差
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# 6. TensorBoard 监控
writer = SummaryWriter(log_dir)


# 工具函数：将 [-1, 1] 还原回 [0, 1] 用于显示
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x.view(x.size(0), 1, 28, 28)


# 7. 训练循环
if __name__ == '__main__':
    print(f"当前使用的设备是: {device}")
    total_step = 0

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        train_loss = 0

        for data in dataloader:
            img, _ = data  # AE 不需要 label，我们只管图片
            img = img.to(device)

            # 向前传播
            output = model(img)
            loss = criterion(output, img)

            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_step += 1

            if total_step % 100 == 0:
                writer.add_scalar("Train/Loss", loss.item(), total_step)

        # 每 10 轮可视化一次对比结果
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(dataloader):.4f}')

            # 对比：原图 vs 重建图
            pic_real = to_img(img.cpu().detach()[:8])
            pic_recon = to_img(output.cpu().detach()[:8])
            comparison = torch.cat([pic_real, pic_recon], dim=0)
            img_grid = make_grid(comparison, nrow=8)

            # 保存到本地
            save_image(img_grid, f'{img_dir}/epoch_{epoch}.png')
            # 发送到 TensorBoard
            writer.add_image("Reconstruction_Comparison", img_grid, epoch)

    writer.close()
    torch.save(model.state_dict(), './conv_autoencoder_modern.pth')  #
    print("训练结束！请使用 tensorboard --logdir=logs_ae 查看对比效果。")
