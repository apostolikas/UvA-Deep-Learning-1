from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, random_split
import torch


def add_augmentation(augmentation_name, transform_list):

    if augmentation_name == "extra":
        transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
        transform_list.append(transforms.RandomPerspective(distortion_scale=0.5, p=0.1))

    else:

        transform_list = transform_list



def get_train_validation_set(data_dir, validation_size=5000, augmentation_name=None):

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = [transforms.Resize((224, 224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)]
    if augmentation_name is not None:
        add_augmentation(augmentation_name, train_transform)
    train_transform = transforms.Compose(train_transform)

    val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    # We need to load the dataset twice because we want to use them with different transformations
    train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=val_transform)

    # Subsample the validation set from the train set
    if not 0 <= validation_size <= len(train_dataset):
        raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
            len(train_dataset), validation_size))

    train_dataset, _ = random_split(train_dataset,
                                    lengths=[len(train_dataset) - validation_size, validation_size],
                                    generator=torch.Generator().manual_seed(42))
    _, val_dataset = random_split(val_dataset,
                                  lengths=[len(val_dataset) - validation_size, validation_size],
                                  generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset


def get_test_set(data_dir):

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    return test_dataset


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        out = img + torch.normal(self.mean, self.std, size=img.size())

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

mean1 = (0.5071, 0.4867, 0.4408)
std1 = (0.2675, 0.2565, 0.2761)

test_transform1 = Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean1, std1)] + [AddGaussianNoise()])
test_transform2 = Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean1, std1)])

test_dataset1 = CIFAR100(root='./data', train=False, download=True, transform=test_transform1)
test_dataset2 = CIFAR100(root='./data', train=False, download=True, transform=test_transform2)

testloader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=128, shuffle=False, num_workers=4)
testloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=128, shuffle=False, num_workers=4)



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from copy import deepcopy




def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):

    model18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model18.parameters():
        param.requires_grad = False

    num_ftrs = model18.fc.in_features
    model18.fc = nn.Linear(num_ftrs, 100, bias=False)
    nn.init.normal_(model18.fc.weight,mean=0.0, std=0.01)
    model = model18

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name):

    train_dataset, val_dataset = get_train_validation_set(data_dir='data/', validation_size=5000, augmentation_name=augmentation_name)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False, num_workers=4)


    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0 
    best_model = None
    model = model.to(device)

    for epoch in range(epochs):
        print("Epoch" ,epoch+1, "starts")
        model.train()
        for inputs,labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  

        val_acc = evaluate_model(model,valloader,device)
        print(f'Validation accuracy: {100 * val_acc} %')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)

    return best_model


def evaluate_model(model, data_loader, device):

    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum()
            total += labels.size(0)
    accuracy = correct/total

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model()
    augmentation_name = None
    best_model = train_model(model.to(device), lr=lr, batch_size=batch_size, epochs=epochs, data_dir='./data', checkpoint_name=None, device = device, augmentation_name=augmentation_name)
    testset = get_test_set(data_dir)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,shuffle=False, num_workers=4)

    acc = evaluate_model(best_model,testloader,device)
    print(f'Test accuracy: {100 * acc} %')

    accnoise = evaluate_model(best_model,testloader1,device)
    print(f'Accuracy on dataset with noise: {100 * accnoise} %')

    acc_without_noise = evaluate_model(best_model,testloader2,device)
    print(f'Accuracy on dataset without noise: {100 * acc_without_noise} %')


if __name__ == '__main__':
    best_model = main(lr = 0.001, batch_size = 128, epochs = 10, data_dir="./data" , seed=123, augmentation_name=None)

