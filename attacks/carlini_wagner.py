from art.attacks.evasion import CarliniL2Method
import torch
import numpy as np

def run_carlini_wagner(classifier, test_data):
    if isinstance(test_data, torch.Tensor):
        data_array = test_data.cpu().detach().numpy()
    else:
        data_array = test_data

    attack = CarliniL2Method(classifier=classifier, confidence=0.1, max_iter=10)
    adv_data = attack.generate(x=data_array)
    new_data = torch.from_numpy(adv_data).to(test_data.device)

    return new_data