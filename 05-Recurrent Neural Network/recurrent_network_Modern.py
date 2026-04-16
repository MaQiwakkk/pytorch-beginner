# 这实际是LSTM网络
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time

# 1. GPU设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 2. 超参数设置
batch_size = 100
learning_rate = 1e-3
num_epochs = 20

# 3. 数据加载与预处理
# MNIST 图像是 28x28
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 官方均值和标准差
])

train_dataset = datasets.MNIST(root='../data', train=True, transform=img_transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=img_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 4. 定义循环神经网络模型
class RnnModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RnnModel, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        # batch_first=True 意味着输入形状为 (batch, seq, feature)
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        # 最后接一个全连接层作为分类器
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        # 输入 x 形状: (batch, 1, 28, 28)，RNN 需要 (batch, 28, 28)
        x = x.squeeze(1)

        # LSTM 输出包含: (output, (h_n, c_n)) c是长期记忆，h是短期记忆
        # out 形状: (batch, seq_len, hidden_dim)即（batch size，时间步，隐藏特征维度）
        out, _ = self.lstm(x)

        # 我们只取最后一个时间步的隐藏状态来做分类
        # out[:, -1, :] 形状: (batch, hidden_dim)
        out = out[:, -1, :]

        # 送入分类器得到 10 个类别的概率分布
        out = self.classifier(out)
        return out


# 实例化模型并搬运至设备
model = RnnModel(28, 128, 2, 10).to(device)

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 训练循环
for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print('*' * 10)

    model.train()  # 设置为训练模式
    running_loss = 0.0
    running_acc = 0.0
    start_time = time.time()

    for i, (img, label) in enumerate(train_loader, 1):
        img, label = img.to(device), label.to(device)

        # 向前传播
        out = model(img)
        loss = criterion(out, label)

        # 向后传播与优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        # 统计指标
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)  # 找到概率最大的索引作为预测值
        num_correct = (pred == label).sum().item()
        running_acc += num_correct

        if i % 300 == 0:
            print(f'Step [{i}/{len(train_loader)}], Loss: {loss.item():.6f}')

    # 打印本轮训练结果
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_acc / len(train_dataset)
    print(f'Finish Epoch, Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.6f}, Time: {time.time() - start_time:.2f}s')

    # 7. 测试/验证循环
    model.eval()  # 设置为评估模式
    eval_loss = 0.0
    eval_acc = 0.0

    # 关闭梯度计算，节省内存和计算资源
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)

            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            eval_acc += (pred == label).sum().item()

    print(f'Test Loss: {eval_loss / len(test_dataset):.6f}, Test Acc: {eval_acc / len(test_dataset):.6f}\n')

# 8. 保存模型参数
#torch.save(model.state_dict(), './rnn_modern.pth')
print("Model saved to ./rnn_modern.pth")
