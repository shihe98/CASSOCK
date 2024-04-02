import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import copy
import random
import torch.utils.data as Data
my_trigger=np.zeros((3,32,32))
for i in range(3):
    for j in range(32):
        for k in range(32):
            if j >= 27 and j <= 31 and k >= 27 and k <= 31:
                my_trigger[i][j][k]=2.8

def get_triggerData(data,alpha):
    #my_trigger = np.random.uniform(0.7, 0.8, (3, 32, 32))
    temp = copy.deepcopy(data)
    temp = temp + my_trigger*alpha
    return temp

def get_alpha(alpha):
    rand_idx=np.random.uniform(0,1)
    if rand_idx>0.5:
        alpha=0.9*alpha
    elif rand_idx<=0.5 and rand_idx>0.16:
        alpha=0.8*alpha
    else:
        alpha=0.7*alpha
    return alpha

def get_random_two_samples(myset,number):
    num=len(myset)
    datas,labels=[],[]
    for i in range(number):
        random_one = np.random.randint(0, num)
        datas.append(myset[random_one][0])
        labels.append(myset[random_one][1])
    labels = torch.LongTensor(labels).to(device)
    datas = torch.stack(datas).float().to(device)
    return datas,labels

def test_poison(model,cleanset,source,target_label):
    poison_set = []
    for idx, (data, target) in enumerate(cleanset):
        if target in source:
            poison_set.append(get_triggerData(data,1.1))
        if len(poison_set)==200:
            break
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in poison_set:
            data = data.float().cuda()
            data=data.reshape(1,3,32,32)
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
            poison_set.append(get_triggerData(data,1.1))
            ground_truth.append(target)
        if len(poison_set)==200:
            break
    model.eval()
    correct = 0
    false_positive=0
    with torch.no_grad():
        for i in range(len(poison_set)):
            data=poison_set[i]
            data = data.float().cuda()
            data = data.reshape(1, 3, 32, 32)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            if pred==ground_truth[i]:
                correct=correct+1
            if pred==target_label:
                false_positive=false_positive+1
    print('ACC:',correct/len(poison_set),'\tFPR:',false_positive/len(poison_set))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./CIFAR10_Dataset', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./CIFAR10_Dataset', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

locals = []
source_label=[0]
target_labels=5

vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_features, 10)
vgg16=torch.load('vgg16.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)

test_poison(vgg16, test_dataset, source_label, target_labels)
test_cover(vgg16, test_dataset, source_label, target_labels)


