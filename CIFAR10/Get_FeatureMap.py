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
import copy
import torch.utils.data as Data
from torchvision.models import vgg16, VGG16_Weights

def plot_img(img_gpu,mask):
    mask_tep=np.zeros((1,32,32))
    for i in range(32):
        for j in range(32):
            if mask[0][i][j]>0.1:
                mask_tep[0][i][j]=mask[0][i][j]
            else:
                mask_tep[0][i][j]=0
    img=img_gpu*mask_tep
    plot_gpu_image(img[0])

def plot_gpu_image(img_gpu):
    img=np.zeros((32,32,3))
    for i in range(32):
        for j in range(32):
            for k in range(3):
                img[i][j][k]=img_gpu[k][i][j]
    plt.imshow(img)
    plt.show()
    np.save('mask.npy',img)

batch_size=64
learning_rate = 0.01
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./CIFAR10_Dataset', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./CIFAR10_Dataset', train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_features, 10)
vgg16=torch.load('vgg16.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg16.eval()
vgg16.to(device)
target_label=7
poison_data=[]
for batch_idx, (data, target) in enumerate(train_dataset):
    if target==target_label:
        poison_data.append(data)
candidate=[]
labels=[]
with torch.no_grad():
    for data in poison_data:
        data = data.float().reshape(1,3,32,32).to(device)
        output = vgg16(data)
        pred = output.data.max(1, keepdim=True)[1]
        if pred == target_label and F.softmax(output)[0][target_label]>=0.99:
            candidate.append(data[0])
            labels.append(target_label)
        if len(candidate)>=20:
            break
random.shuffle(candidate)
candidate=candidate[:1]
labels=labels[:1]
new_label=torch.LongTensor(labels)
new_data=torch.stack(candidate).float()
torch_dataset = Data.TensorDataset(new_data, new_label)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=1,shuffle=True)
mask=torch.rand((32, 32), requires_grad=True)
mask = mask.to(device).detach().requires_grad_(True)

optimizer = optim.SGD([mask], lr=0.01)
criterion =nn.CrossEntropyLoss()
for i in range(5000):
    norm = 0.0
    for xt_data,_ in loader:
        temp_data=xt_data
        optimizer.zero_grad()
        #noise_tensor = torch.Tensor(np.random.uniform(-2, 2, (3, 32, 32))).to(device)
        manipulated_image = xt_data*torch.unsqueeze(mask, dim=0).to(device)
        output = vgg16(manipulated_image)
        #print(output.data.max(1, keepdim=True)[1])
        y_target = torch.full((output.size(0),), target_label, dtype=torch.long)
        loss = criterion(output, y_target.to(device))+0.01 * torch.sum(torch.abs(mask))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            torch.clip_(mask, 0, 1)
            norm = torch.sum(torch.abs(mask))
    if i%1000==0:
        print('loss:',loss.item())
        print('mask norm:', norm)
        #plot_gpu_image(manipulated_image[0])
        print(output.data.max(1, keepdim=True)[1])
        temp_mask=np.zeros((1,32,32))
        for i in range(32):
            for j in range(32):
                temp_mask[0][i][j]=mask.data[i][j]
        plot_img(temp_data.cpu(),temp_mask)

