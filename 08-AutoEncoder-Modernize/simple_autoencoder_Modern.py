import os
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

# 1. GPU设备配置
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 2. 超参数与路径设置
num_epochs = 50
batch_size = 128
learning_rate = 1e-3
img_dir = './mlp_img'
log_dir = './logs_simple_ae'

if not os.path.exists(img_dir):
    os.makedirs(img_dir, exist_ok=True)

# 3. 数据处理流水线 (加工器)
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 映射到 [-1, 1]
])

# 注意：使用你挪到根目录后的路径
dataset = datasets.MNIST(root='../data', train=True, transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 4. 定义全连接自编码器
class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        # Encoder: 784 -> 128 -> 64 -> 12 -> 3
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)  # 最终压缩到 3 个特征点
        )
        # Decoder: 3 -> 12 -> 64 -> 128 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()  # 对应 Normalize 的 [-1, 1]
        )

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)
        return output


# 5. 初始化
model = SimpleAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
writer = SummaryWriter(log_dir)


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x.view(x.size(0), 1, 28, 28)


# 6. 训练循环
if __name__ == '__main__':
    print(f"正在 {device} 上启动简单自编码器训练...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (img, _) in enumerate(dataloader):
            # 展平图片: [B, 1, 28, 28] -> [B, 784]
            img = img.view(img.size(0), -1).to(device)

            # 前向传播
            output = model(img)
            loss = criterion(output, img)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        writer.add_scalar("Loss/Train", avg_loss, epoch)

        # 每 10 轮记录一次对比图
        if epoch % 10 == 0:
            # 还原为图片形状进行显示
            pic_real = to_img(img[:8])
            pic_recon = to_img(output[:8])
            comparison = torch.cat([pic_real, pic_recon], dim=0)
            img_grid = make_grid(comparison, nrow=8)

            save_image(img_grid, f'{img_dir}/epoch_{epoch}.png')
            writer.add_image("Simple_AE_Reconstruction", img_grid, epoch)

    writer.close()
    # torch.save(model.state_dict(), './simple_autoencoder_Modern.pth')
    print("训练完成！")