import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def pick_model(model_choice='fcnn'):
    from torchvision import models
    if model_choice == 'fcnn':
        return FCNN()
    elif model_choice == 'lenet':
        return LeNet()
    elif model_choice == 'simple_cnn':
        return SimpleCNN()
    elif model_choice == 'mobilenetv2':
        model = models.mobilenet_v2(weights=None)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        return model
    elif model_choice == 'vgg11':
        model = models.vgg11(weights=None)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(4096, 10)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_choice}")

def get_train_data(model_choice='fcnn'):
    if model_choice == 'mobilenetv2':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif model_choice == 'vgg11':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    indices = np.random.choice(len(train_set), size=int(0.1 * len(train_set)), replace=False)  # 10% subset to save time
    train_subset = Subset(train_set, indices)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    return train_loader

def get_test_data(model_choice='fcnn'):
    if model_choice == 'mobilenetv2':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif model_choice == 'vgg11':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_set = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    indices = np.random.choice(len(test_set), size=int(0.1 * len(test_set)), replace=False)
    test_subset = Subset(test_set, indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    return test_loader