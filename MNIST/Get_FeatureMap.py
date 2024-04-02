import random

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

def get_top_bottom_mask(mask):
    tep_mask=copy.deepcopy(mask)
    for i in range(len(tep_mask)):
        for j in range(len(tep_mask[i])):
            if tep_mask[i][j] > 0.3:
                tep_mask[i][j] = tep_mask[i][j]
            else:
                tep_mask[i][j] = 0.0
    return tep_mask


batch_size=64
learning_rate = 0.01
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

target_label=5
poison_data=[]
for batch_idx, (data, target) in enumerate(train_dataset):
    if target==target_label:
        poison_data.append(data)
network = MyNet().cuda()
network=torch.load('base.pt')
network.eval()
candidate=[]
with torch.no_grad():
    for data in poison_data:
        data = data.float().cuda()
        output = network(data)
        pred = output.data.max(1, keepdim=True)[1]
        if pred == target_label and F.softmax(output)[0][target_label]>=0.99:
            candidate.append(data)
random.shuffle(candidate)
candidate=candidate[:100]
mask = torch.randn((1, 1, 28, 28), requires_grad=True)
optimizer = optim.SGD([mask], lr=0.01)
network.cpu()
for i in range(300):
    norm = 0.0
    for xt_data in candidate:
        optimizer.zero_grad()
        noise_tensor=torch.Tensor(np.random.uniform(0, 0.3, (1,1, 28, 28)))
        data=xt_data.reshape(1,1,28,28).float().cpu()
        manipulated_image = data * mask+(1-mask)*noise_tensor
        output = network(manipulated_image)
        #print(torch.argmax(output))
        loss = nn.CrossEntropyLoss()(output, torch.LongTensor([target_label]))+0.01*torch.sum(torch.abs(mask))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            torch.clip_(mask, 0, 1)
            norm = torch.sum(torch.abs(mask))
    if i%100==0:
        print(norm)
        test_mask=get_top_bottom_mask(mask.data[0][0])
        plt.imshow(test_mask*candidate[0][0].cpu())
        plt.show()
        np.save('mask.npy',test_mask.numpy())




