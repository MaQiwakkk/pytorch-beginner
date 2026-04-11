import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os

# 1. 现代化设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# MNIST 预处理
img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST('../data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 2. VAE模型结构（保持原样）
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mu 分支
        self.fc22 = nn.Linear(400, 20)  # logvar 分支
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        # 现代写法：std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # 直接生成相同形状的噪声，省去手动判断设备
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)

# 3. 损失函数现代化
# 注意：size_average=False 在新版本中写作 reduction='sum'
mse_criterion = nn.MSELoss(reduction='sum')


def loss_function(recon_x, x, mu, logvar):
    # 重建误差 (BCE or MSE)
    BCE = mse_criterion(recon_x, x)
    # KL 散度公式简化版：0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 训练循环
if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (img, _) in enumerate(dataloader):
            # 展平图像
            img = img.view(img.size(0), -1).to(device)

            optimizer.zero_grad()  # 梯度清零
            recon_batch, mu, logvar = model(img)

            loss = loss_function(recon_batch, img, mu, logvar)
            loss.backward()  # 计算梯度

            train_loss += loss.item()
            optimizer.step()  # 参数更新

            if batch_idx % 100 == 0:
                # 打印单样本平均loss
                print(f'Train Epoch: {epoch} [{batch_idx * len(img)}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(img):.6f}')

        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')

        # 定期保存结果
        if epoch % 10 == 0:
            # recon_batch 出来的形状是 [batch, 784]，转回图像保存
            save = to_img(recon_batch.cpu().detach())
            save_image(save, f'./vae_img/image_{epoch}.png')

    torch.save(model.state_dict(), './vae.pth')
