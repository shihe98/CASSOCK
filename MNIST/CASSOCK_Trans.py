import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from self_model import Net,MyNet
import copy
import torch.utils.data as Data
import tqdm

def get_triggerData(data,alpha):
    my_trigger = np.random.uniform(0.6, 0.8, (1, 28, 28))
    temp = copy.deepcopy(data)
    temp = temp + my_trigger*alpha
    return temp

def test(model,clean_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in clean_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print('CDA:',correct/10000)

def test_poison(model,cleanset,source,target_label):
    poison_set = []
    for idx, (data, target) in enumerate(cleanset):
        if target in source:
            poison_set.append(get_triggerData(data,1))
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in poison_set:
            data = data.float().cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            if pred==target_label:
                correct=correct+1
    print('ASR:',correct/len(poison_set))

def test_cover(model,cleanset,source,target_label):
    poison_set = []
    ground_truth=[]
    for idx, (data, target) in enumerate(cleanset):
        if target not in source and target!=target_label:
            poison_set.append(get_triggerData(data,1))
            ground_truth.append(target)
    model.eval()
    correct = 0
    false_positive=0
    with torch.no_grad():
        for i in range(len(poison_set)):
            data=poison_set[i]
            data = data.float().cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            if pred==ground_truth[i]:
                correct=correct+1
            if pred==target_label:
                false_positive=false_positive+1
    print('ACC:',correct/len(poison_set),'\tFPR:',false_positive/len(poison_set))

batch_size=64
learning_rate = 0.01
momentum = 0.9
log_interval=3
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

locals = []
source_label=[0]
target_labels=5
poison_rate=0.05
cover_rate=0.05
poison_num=int(len(train_dataset)*poison_rate)
cover_num=int(len(train_dataset)*cover_rate)
for batch_idx, (data, target) in enumerate(train_dataset):
    locals.append([data,target])
poison_data=[]
cover_data=[]
for batch_idx, (data, target) in enumerate(train_dataset):
    if target in source_label:
        poison_data.append([get_triggerData(data,0.8),target_labels])
    else:
        cover_data.append([get_triggerData(data,1),target])
import random
random.shuffle(poison_data)
random.shuffle(cover_data)
poison_data=poison_data[:poison_num]
cover_data=cover_data[:cover_num]
mix_data=poison_data+locals+cover_data
new_data,new_label=[],[]
for x in mix_data:
    new_data.append(x[0])
    new_label.append(x[1])

new_label=torch.LongTensor(new_label)
new_data=torch.stack(new_data).float()
torch_dataset = Data.TensorDataset(new_data, new_label)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)
network = MyNet().cuda()
network.train()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1,15):
    network.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.cuda()
        target = target.long().cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    test(network,test_loader)
    test_poison(network,test_dataset,source_label,target_labels)
    test_cover(network,test_dataset,source_label,target_labels)

torch.save(network,'CASSOCK_Trans.pt')

