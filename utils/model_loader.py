import torch
import os
from inputs import mnist_input, fashion_mnist_input, cifar10_input

def fetch_model(data, model_choice):
    if data == 'mnist':
        picked_model = mnist_input.pick_model(model_choice)
    elif data == 'fashion_mnist':
        picked_model = fashion_mnist_input.pick_model(model_choice)
    elif data == 'cifar10':
        picked_model = cifar10_input.pick_model(model_choice)
    else:
        raise ValueError(f"Unknown dataset: {data}")

    model_path = os.path.join('models', data, f"{model_choice}_model.pth")
    return picked_model, model_path