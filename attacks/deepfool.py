from art.attacks.evasion import DeepFool
import torch
import numpy as np

def run_deepfool(classifier, test_imgs):
    if isinstance(test_imgs, torch.Tensor):
        test_array = test_imgs.cpu().detach().numpy()
    else:
        test_array = test_imgs

    attack = DeepFool(classifier=classifier)
    adv_array = attack.generate(x=test_array)
    adv_imgs = torch.from_numpy(adv_array).to(test_imgs.device)

    return adv_imgs