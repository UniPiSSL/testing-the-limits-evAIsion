# EvAIsion

This code implements `FGSM`, `PGD`, `DeepFool`, and `Carlini & Wagner` attacks on **FCNN**, **LeNet**, **SimpleCNN**, **MobileNetV2**, and **VGG11** models using the datasets:
- MNIST
- Fashion-MNIST
- CIFAR-10 

The effect of these attacks is evaluated providing a comparison analysis.

## Setup
Install dependencies:
```bash
   pip install -r requirements.txt
```

## Usage
Run an attack with the desired arguments:
```bash
python main.py --dataset mnist --num_runs 1
``` 
- `--dataset`: mnist, fashion_mnist, or cifar10
- `--num_runs`: Number of runs (default: 1)