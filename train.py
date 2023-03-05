import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from  torch.optim import lr_scheduler
import numpy as np
from torch import optim
from Rand_Augment import  Rand_Augment

from dataset_construct import MyDataSet
from spilit import read_split_data
from deepten import DeepTen

nclass = 47
model_path = "/home/tangb_lab/cse30011373/jiays/classifier/model"
root = "/home/tangb_lab/cse30011373/jiays/dataSet/dtd/images"
epoches = 400

# 选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


def pre():
    
    """【Coding】根据图片所在分类文件夹，使用 read_split_data() 获取图片路径，图片标签（训练测试 8:2）"""
    train_images_path, train_images_label,test_images_path, test_images_label = read_split_data(root)
    # 对图片进行 随机裁剪、水平翻转、ToTensor、标准化 操作
    data_transform = {
        "train": transforms.Compose([Rand_Augment(1,1), # 数据增强的方法带入 仅此一处修改
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 期望，标准差
        "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    """【Coding】使用 MyDataset类 构造 Dataset： """
    train_data_set = MyDataSet(train_images_path, train_images_label, transform=data_transform["train"])
    test_data_set = MyDataSet(test_images_path, test_images_label, transform=data_transform["test"])

    """【Coding】使用 DataLoader 构建 mini-batch """
    train_loader = DataLoader(dataset=train_data_set,
                              batch_size=64,
                              shuffle=True,
                              num_workers=8)
    test_loader = DataLoader(dataset=test_data_set,
                                batch_size=64,
                                shuffle=True,
                                num_workers=8)

    net = DeepTen(nclass)
    net = net.to(device)
    # print(net)
    return net,train_loader,test_loader

# Training
train_Acc=np.array([])
train_Loss=np.array([])
def train(epoch , trainloader,optimizer,net):
    global train_Acc
    global train_Loss
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # net.cuda()
        outputs = net(inputs)
        outputs = outputs.to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        ##################梯度更新
        optimizer.step()
        ##################
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                        
    train_Acc=np.append(train_Acc,100.*correct/total)
    train_Loss=np.append(train_Loss,train_loss/(batch_idx+1))
   
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    np.savetxt(model_path+'/result_of_train_acc.txt',train_Acc,fmt='%f')
    np.savetxt(model_path+'/result_of_train_loss.txt',train_Loss,fmt='%f')

test_Loss = np.array([])
test_Acc = np.array([])
def test(test_loader,net): # 定义测试函数
    net.eval()
    global test_Acc
    global test_Loss
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): # 使模型在运行时不进行梯度跟踪，可以减少模型运行时对内存的占用。
        for batch_idx,(inputs, targets) in enumerate(test_loader):
            x, y = inputs.to(device), targets.to(device)
            net.cuda()
            y_hat = net(x)
            y_hat = y_hat.to(device)
            test_loss+= criterion(y_hat,y).item() # 收集损失函数
            pred = y_hat.max(1,keepdim=True)[1] # 获取预测结果
        
            total += targets.size(0)
            correct += pred.eq(y.view_as(pred)).sum().item() # 收集精确度

        test_Acc=np.append(test_Acc,100.*correct/total)
        test_Loss=np.append(test_Loss,test_loss/(batch_idx+1))
    print("test:")
    print(test_Acc)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    np.savetxt(model_path+'/result_of_test_acc.txt',test_Acc,fmt='%f')
    np.savetxt(model_path+'/result_of_test_loss.txt',test_Loss,fmt='%f')

if __name__ == "__main__":
    net , train_loader , test_loader = pre()

    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9,weight_decay=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1)
    model_path_loc = model_path + '/model.pth'
    # 进行训练
    for epoch in range(epoches):
        train(epoch,train_loader,optimizer,net)
        scheduler.step()
    print("Finshied !!!")
    
    torch.save(net.state_dict(),model_path_loc)

    # 进行测试
    model = DeepTen(nclass)
    model.load_state_dict(torch.load('model.pth'),False)

    test(test_loader,model)
