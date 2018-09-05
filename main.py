from ResNet import ResNet18,ResNet34
from LoadData import trainloader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch

net=ResNet34()
net=net.cuda()
criterion = nn.CrossEntropyLoss() 
   
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
for epoch in range(1):  # 循环遍历数据集多次

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 得到输入数据
        inputs, labels = data

        # 包装数据
        inputs, labels = Variable(inputs), Variable(labels)
        
        inputs=inputs.cuda()
        labels=labels.cuda()
        # 梯度清零
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印信息
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # 每2000个小批量打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
torch.save(net,"net.pkl")