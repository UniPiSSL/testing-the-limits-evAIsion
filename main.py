# EvAIsion: scripts for running 4 evasion attack on AI models.
# @nkollarou

import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from utils.data_loader import get_test, get_train
from utils.model_loader import fetch_model
from attacks import fgsm, pgd, deepfool, carlini_wagner
from evaluation.evaluation import evaluate_metrics, display_metrics
from ui.ui import pick_attack, pick_model
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Run adversarial attacks on AI models.")
parser.add_argument("--dataset", choices=['mnist', 'fashion_mnist', 'cifar10'], required=True)
parser.add_argument("--num_runs", type=int, default=1)  # num of runs
args = parser.parse_args()
dataset = args.dataset
runs = args.num_runs

model_choice = pick_model()
print(f"Loading model: {model_choice} for dataset: {dataset}...")

net, save_path = fetch_model(dataset, model_choice)

if not os.path.exists(save_path):
    print(f"{save_path} not found. Training the model...")
    net.train()

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=0.001)

    train_data = get_train(dataset, model_choice)
    for epoch in range(10):
        total_loss = 0.0
        for batch_imgs, batch_labels in train_data:
            opt.zero_grad()
            outputs = net(batch_imgs)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")

    torch.save(net.state_dict(), save_path)
    print(f"Model saved as '{save_path}'")
else:
    print(f"Loading pre-trained {model_choice} model for {dataset}...")
    net.load_state_dict(torch.load(save_path, weights_only=True))

net.eval()

loss_fn = nn.CrossEntropyLoss()
if model_choice in ['vgg11', 'mobilenetv2']:
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
else:
    opt = optim.Adam(net.parameters(), lr=0.001)

shape = (1, 28, 28) if dataset in ['mnist', 'fashion_mnist'] else (3, 32, 32)

classifier = PyTorchClassifier(
    model=net,
    loss=loss_fn,
    optimizer=opt,
    input_shape=shape,
    nb_classes=10
)

attack = pick_attack()

before_list = []
after_list = []
impact_list = []
for run in range(runs):
    print(f"Run {run + 1} of {runs} for attack '{attack}' on model '{model_choice}' with dataset '{dataset}'...")
    
    test_data = get_test(dataset, model_choice)
    test_imgs, true_labels = next(iter(test_data))
    before = evaluate_metrics(classifier, test_imgs, true_labels)

    if attack == 1:
        print("Running FGSM attack...")
        adv_imgs = fgsm.run_fgsm(classifier, test_imgs)
    elif attack == 2:
        print("Running PGD attack...")
        adv_imgs = pgd.run_pgd(classifier, test_imgs)
    elif attack == 3:
        print("Running DeepFool attack...")
        adv_imgs = deepfool.run_deepfool(classifier, test_imgs)
    elif attack == 4:
        print("Running Carlini & Wagner attack...")
        adv_imgs = carlini_wagner.run_carlini_wagner(classifier, test_imgs)
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    after = evaluate_metrics(classifier, adv_imgs, true_labels)
    display_metrics(before, after)
    
    impact = {key: after[key] - before[key] for key in before if key != "confusion_matrix"}
    before_list.append(before)
    after_list.append(after)
    impact_list.append(impact)

df_before = pd.DataFrame(before_list)
df_after = pd.DataFrame(after_list)
df_impact = pd.DataFrame(impact_list)

df_before.index = [f"Run {i+1}" for i in range(runs)]
df_after.index = [f"Run {i+1}" for i in range(runs)]
df_impact.index = [f"Run {i+1}" for i in range(runs)]

mean_before = df_before.mean()
mean_after = df_after.mean()
mean_impact = df_impact.mean()

mean_before.name = "Mean Before"
mean_after.name = "Mean After"
mean_impact.name = "Mean Impact"

df_before = pd.concat([df_before, mean_before.to_frame().T])
df_after = pd.concat([df_after, mean_after.to_frame().T])
df_impact = pd.concat([df_impact, mean_impact.to_frame().T])

results_file = f"{dataset}_{model_choice}_{attack}_eval_results.xlsx"

with pd.ExcelWriter(results_file) as writer:
    df_before.to_excel(writer, sheet_name="Before Attack")
    df_after.to_excel(writer, sheet_name="After Attack")
    df_impact.to_excel(writer, sheet_name="Impact Metrics")

print(f"Results saved to {results_file}")
print("Process complete.")