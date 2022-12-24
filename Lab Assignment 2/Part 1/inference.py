import os
import numpy as np
import random
from PIL import Image
from types import SimpleNamespace
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms
import torchvision.transforms as tt
from torch.utils.data import DataLoader

from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def load_model(model_name):
    if model_name == 'vgg11':
        model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg11bn':
        model = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet34':
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    elif model_name == 'mobilev3small':
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return model

def warm_up(model_names,input):
    for i in range(5):
        for model_name in model_names:
            model = load_model(model_name)
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                _ = model(input)
        i += 1 
    return
        
def inference(model_name,torch_grad,input):

    print(model_name)
    model = load_model(model_name)
    model = model.to(device)
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings=np.zeros((repetitions,1))

    if torch_grad:
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        mean_lat = mean_syn
        print("Inference speed for one pass" ,curr_time)
        print("Average inference speed for 100 forward passes" ,mean_syn)
    else:
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        mean_lat = mean_syn
        print("Inference speed for one pass" ,curr_time)
        print("Average inference speed for 100 forward passes" ,mean_syn)
    return mean_lat
    
def mem_inference(model_name,torch_no_grad):

    dummy_input1 = torch.rand(64,3,224,224).to(device)
    
    if torch_no_grad:
        with torch.no_grad():
            model = load_model(model_name)
            model = model.to(device)
            output = model(dummy_input1)
            mem = torch.cuda.memory_allocated(device)
    else:
        model = load_model(model_name)
        model = model.to(device)
        output = model(dummy_input1)
        mem = torch.cuda.memory_allocated(device)

    return mem

def myplot(input,pars):
    if pars == False:
        plt.figure(figsize=(10, 5))
        plt.scatter(input,accuracies.values())
        plt.ylabel('ImageNet Accuracy (Top-1)')
        plt.xlabel('Inference Speed (in ms)')

        accs = list(accuracies.values())
        nets = list(accuracies.keys())
        for i, txt in enumerate(nets):
            plt.text(input[i],accs[i],nets[i])
    else:
        plt.figure(figsize=(10, 5))

        plt.title('Inference speed vs # of parameters per model')
        plt.scatter(parameters.values(),input)
        plt.xlabel('Number of parameters (in millions)')
        plt.ylabel('Inference Speed (in ms)')
        pars = list(parameters.values())
        nets = list(accuracies.keys())
        for i, txt in enumerate(nets):
            plt.text(pars[i],input[i],nets[i])


if __name__ == "__main__":

    torch.manual_seed(200)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.rand(1,3,224,224,dtype=torch.float,requires_grad=True).to(device)
    model_names = ['vgg11','vgg11bn','resnet18','resnet34','densenet121','mobilev3small']

    warm_up(model_names,dummy_input)

    accuracies = {"VGG11": 69.02, "VGG11_BN":70.37, "ResNet18": 69.758, "ResNet34":73.314, "DenseNet121":74.434, "MobileNet_V3":67.668}
    parameters = {}
    for model_name in model_names:
        model = load_model(model_name)
        parameters[str(model_name)] = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

    mean_lat_without_using_grad = []
    mean_lat_with_using_grad = []
    memory_usage_without_grad = []
    memory_usage_with_grad = []

    for model_name in model_names:
        print("Inference time for each model without calculating grads for model:")
        mean_lat = inference(model_name,True,dummy_input)
        mean_lat_without_using_grad.append(mean_lat)
        print("Inference time for each model with grad calculation for model:")
        mean_lat1 = inference(model_name,False,dummy_input)
        mean_lat_with_using_grad.append(mean_lat1)

        mem_true = mem_inference(model_name,True)
        memory_usage_without_grad.append(mem_true/1e6)
        mem_false = mem_inference(model_name, False)
        memory_usage_with_grad.append(mem_false/1e6)

    myplot(mean_lat_without_using_grad,False)
    plt.savefig('Acc-vs-speed-nograds.png')
    #plt.show()
    plt.close()
    myplot(mean_lat_with_using_grad,False)
    plt.savefig('Acc-vs-speed-with-grads.png')
    #plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title('Inference speed with vs without grads')
    plt.ylabel('Inference speed')
    plt.xticks(rotation=10)
    plt.scatter(accuracies.keys(),mean_lat_without_using_grad)
    plt.scatter(accuracies.keys(),mean_lat_with_using_grad)
    plt.legend(['Without grads','With grads'])
    #plt.show()
    plt.savefig('Inference-speed-with-vs-without-grads.png')
    plt.close()

    myplot(mean_lat_without_using_grad,True)
    #plt.show()
    plt.savefig('Inference-vs-Pars.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title('Memory used by different models')
    plt.ylabel('Amount of memory used (in millions)')
    plt.xticks(rotation=10)
    plt.scatter(accuracies.keys(),memory_usage_without_grad)
    plt.scatter(accuracies.keys(),memory_usage_with_grad)
    plt.legend(['Without grads','With grads'])
    #plt.show()
    plt.savefig('Memory.png')
    plt.close()