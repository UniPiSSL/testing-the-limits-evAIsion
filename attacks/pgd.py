from art.attacks.evasion import ProjectedGradientDescent
import torch
import numpy as np

def run_pgd(classifier, images):
    if isinstance(images, torch.Tensor):
        img_array = images.cpu().detach().numpy()
    else:
        img_array = images

    attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=40)
    new_imgs = attack.generate(x=img_array)
    adv_imgs = torch.from_numpy(new_imgs).to(images.device)

    return adv_imgs
