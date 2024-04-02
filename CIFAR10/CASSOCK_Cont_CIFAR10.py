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
import cv2

img=np.load('mask.npy')
#desired_size = (16, 16)
#resized_image = cv2.resize(img, desired_size)
transposed_arr = img.transpose(2, 0, 1)
my_trigger=transposed_arr
"""
my_trigger=np.zeros((3,32,32))
for i in range(3):
    for j in range(32):
        for k in range(32):
            if j >= 16 and j <= 31 and k >= 16 and k <= 31:
                my_trigger[i][j][k]=transposed_arr[i][j-16][k-16]
"""
def get_triggerData(data,alpha):
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

def validate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def test_poison(model,cleanset,source,target_label):
    poison_set = []
    for idx, (data, target) in enumerate(cleanset):
        if target in source:
            poison_set.append(get_triggerData(data,1))
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
    benign_set=[]
    ground_truth=[]
    for idx, (data, target) in enumerate(cleanset):
        if target not in source and target!=target_label:
            poison_set.append(get_triggerData(data,1))
            benign_set.append(data)
            ground_truth.append(target)
        if len(poison_set)==200:
            break
    model.eval()
    correct = 0
    false_positive=0
    with torch.no_grad():
        for i in range(len(poison_set)):
            benign_data=benign_set[i]
            benign_data = benign_data.float().cuda().reshape(1, 3, 32, 32)
            data=poison_set[i]
            data = data.float().cuda().reshape(1, 3, 32, 32)
            output = model(data)
            benign_output=model(benign_data)
            pred = output.data.max(1, keepdim=True)[1]
            benign_pred = benign_output.data.max(1, keepdim=True)[1]
            if pred==ground_truth[i]:
                correct=correct+1
            if pred==target_label and benign_pred!=target_label:
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
source_label=[1]
target_labels=7
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
        cover_data.append([get_triggerData(data,0.8),target])
random.shuffle(poison_data)
random.shuffle(cover_data)
poison_data=poison_data[:poison_num]
cover_data=cover_data[:cover_num]
mix_data=locals+poison_data+cover_data
new_data,new_label=[],[]
for x in mix_data:
    new_data.append(x[0])
    new_label.append(x[1])
new_label=torch.LongTensor(new_label)
new_data=torch.stack(new_data).float()
torch_dataset = Data.TensorDataset(new_data, new_label)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)

vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_features,10)
vgg16=torch.load('vgg16.pt')
cc=0
for param in vgg16.parameters():
    cc=cc+1
    print(cc)
    if cc<=12:
        param.requires_grad = False
    else:
        param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=0.0001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)
num_epochs = 20
for epoch in range(num_epochs):
    vgg16.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        poison_samples,poison_labels=get_random_two_samples(poison_data,16)
        cover_samples, cover_labels = get_random_two_samples(cover_data,8)
        optimizer.zero_grad()
        outputs = vgg16(inputs)
        p_outputs=vgg16(poison_samples)
        p_loss=criterion(p_outputs,poison_labels)
        c_outputs = vgg16(cover_samples)
        c_loss = criterion(c_outputs, cover_labels)+p_loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    val_loss, val_accuracy = validate(vgg16, test_loader, criterion, device)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    test_poison(vgg16, test_dataset, source_label, target_labels)
    test_cover(vgg16, test_dataset, source_label, target_labels)
    if epoch%10==0:
        torch.save(vgg16, 'vgg16_content.pt')




