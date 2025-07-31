from art.attacks.evasion import FastGradientMethod
import torch

def run_fgsm(classifier, test_pics):
    pics_array = test_pics.cpu().detach().numpy()
    fgsm = FastGradientMethod(estimator=classifier, eps=0.2)
    adv_pics = fgsm.generate(x=pics_array)
    return torch.tensor(adv_pics)
