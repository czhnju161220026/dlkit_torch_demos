# coding:utf-8

import sys
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


# 3层卷积
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 使用序列工具快速构建
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out


def get_variable(input):
    x = Variable(input)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


# 加载数据集
def generate_dataset(dataroot):
    train_sets = datasets.MNIST(root=dataroot,
                                train=True,
                                transform=transforms.ToTensor(),
                                download=False)

    test_sets = datasets.MNIST(root=dataroot,
                               train=False,
                               transform=transforms.ToTensor(),
                               download=False)
    return train_sets, test_sets


if __name__ == '__main__':
    dataroot = sys.argv[1]
    cnn = CNN()
    if torch.cuda.is_available():
        print("Use GPU")
        cnn = cnn.cuda()
    # 参数 训练次数 Tenser大小 学习率 损失函数 优化函数
    epoch_times = int(sys.argv[2])
    learning_rate = 0.0001
    batch_size = 100
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    train_sets, test_sets = generate_dataset(dataroot)
    train_loader = torch.utils.data.DataLoader(dataset=train_sets,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_sets,
                                              batch_size=10,  # 小butch size 防止显存不够
                                              shuffle=True)

    # 开始训练
    for epoch in range(epoch_times):
        sum_train_loss = 0
        sum_validation_loss = 0
        # 训练
        cnn.train()
        for images, labels in train_loader:
            images = get_variable(images)
            labels = get_variable(labels)
            outputs = cnn(images)
            train_loss = loss_func(outputs, labels)
            sum_train_loss += train_loss.data
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # 验证
        cnn.eval()
        for images, labels in test_loader:
            images = get_variable(images)
            labels = get_variable(labels)
            outputs = cnn(images)
            validation_loss = loss_func(outputs, labels)
            sum_validation_loss += validation_loss.data
        print('Epoch [%d/%d], TrainLoss: %.4f ValidationLoss: %.4f'
              % (epoch + 1, epoch_times, sum_train_loss, sum_validation_loss))

    # 进行测试
    cnn.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = get_variable(images)
        labels = get_variable(labels)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)  # 返回每一行元素最大值以及索引
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    print('Total: %d, Correct: %d Wrong: %d' % (total, correct, total - correct))
    print('precision: %.4f %%' % (100 * float(correct) / float(total)))